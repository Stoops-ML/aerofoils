import torch.nn as nn
import torch
import sys


def calc_padding(image_size, stride, kernel_size):
    """returns padding for convolution such that input image size = output image size"""
    return (stride * (image_size - 1) - image_size + kernel_size) // 2


def calc_image_out_size(image_in_size, kernel_size, padding, stride, num_layers):
    """image size change due to convolutions"""
    image_out_size = image_in_size  # initialise
    for _ in range(num_layers):
        image_out_size = (image_out_size - kernel_size + 2 * padding) // stride + 1
    return image_out_size


class ConvNet(nn.Module):
    def __init__(self, input_size, convolutions, num_channels, hidden_layers, output_size):
        super(ConvNet, self).__init__()

        # initialise
        self.num_channels = num_channels
        self.convolutions = convolutions

        # convolutions
        kernel_size = 3
        stride = 1
        padding = 0  # todo add padding so no reduction in image size (see below)

        # padding = (stride * (image_size - 1) - image_size + kernel_size) // 2  # no reduction in image size

        # input size change due to convolution and pooling
        self.image_size = int(input_size)
        for _ in range(len(convolutions) * 2):  # *2 for convolution AND pooling
            self.image_size = int((self.image_size - kernel_size + 2 * padding) // stride + 1)  # todo will this produce an int without int()?

        def conv(in_channels, out_channels):
            return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding,
                                           padding_mode='reflect'),
                                 nn.BatchNorm1d(out_channels),  # todo is this supposed to be after the relu?
                                 nn.ReLU(),
                                 nn.MaxPool1d(kernel_size, stride, padding))

        self.extractor = nn.Sequential(
            conv(num_channels, convolutions[0]),
            conv(convolutions[0], convolutions[1]),
            conv(convolutions[1], convolutions[2]),

            nn.Conv1d(convolutions[2], convolutions[3], kernel_size, padding=padding, padding_mode='reflect'),
            nn.BatchNorm1d(convolutions[3]),
            nn.AdaptiveAvgPool1d(self.image_size),  # not sure why but people always end with average pool
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(self.image_size * convolutions[-1], hidden_layers[0]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.LeakyReLU(),

            nn.Linear(hidden_layers[0], output_size),
            nn.BatchNorm1d(num_features=num_channels),
        )

    def forward(self, x):
        out = self.extractor(x)
        out = out.view(-1, self.num_channels, self.image_size * self.convolutions[-1])  # -1 for varying batch sizes
        out = self.fully_connected(out)

        ClCd = out[:, :, 0]
        angle = out[:, :, 1]

        return ClCd, angle


class DenseBlock(nn.Module):
    def __init__(self, in_channels, image_size, dense_out_channels, num_convs):
        """make a dense block"""
        super().__init__()
        kernel_size = 3
        stride = 1  # no reduction in image size
        padding = calc_padding(image_size, stride, kernel_size)
        self.num_convs = num_convs

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_features=in_channels)

        def conv_layer(i):  # make a layer in the dense block
            return nn.Sequential(nn.Conv1d(in_channels=in_channels + dense_out_channels * i,
                                           out_channels=dense_out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding))

        self.dense_block = nn.ModuleList([conv_layer(i) for i in range(num_convs)])  # list of layers in dense block

    def forward(self, x):
        out = self.bn(x)  # todo why is batchnorm done here?

        def dense(conv_func, dense_in):
            nonlocal out
            conv = self.relu(conv_func(dense_in))
            out_conv = torch.cat([out, conv], 1)  # concatenate in channel dimension
            dense_out = self.relu(out_conv)
            return dense_out

        for conv in self.dense_block:
            out = dense(conv, out)

        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, st=2, p=0):
        """make a transition layer"""
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool1d(kernel_size=ks, stride=st, padding=p)  # image size reduction

    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)
        return out


class FullyConnected(nn.Module):
    def __init__(self, channels_list, indx, num_channels):
        """make layer in fully connected neural network"""
        super().__init__()

        self.use_relu = False if (len(channels_list) - 2) == indx else True  # don't use ReLU if last layer in fc NN

        self.linear = nn.Linear(channels_list[indx], channels_list[indx+1])
        self.bn = nn.BatchNorm1d(num_features=num_channels)
        if self.use_relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.bn(self.linear(x))
        if self.use_relu:
            out = self.relu(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, image_size, output_size, convolutions, fc_hidden, num_channels, dense_out_channels):
        """create a densely connected convolutional neural network"""
        super().__init__()

        self.image_size = image_size  # original image size
        self.num_channels = num_channels  # number of input channels
        num_convs = len(convolutions) - 1  # number of convolutions in a dense block
        # num_convs equals number of layers of densenet because it defines the inputs to the transition layers
        # therefore no need for another variable defining number of layers of densenet
        self.relu = nn.ReLU()
        self.convolutions = convolutions
        ks, st, p = 3, 2, 0  # properties of transition layer

        # low convolution
        self.lowconv = nn.Conv1d(in_channels=self.num_channels, out_channels=convolutions[0], stride=1, kernel_size=7,
                                 padding=calc_padding(image_size, 1, 7),  # image size unchanged
                                 bias=False)  # TODO: bias terms set to false. Set them to true?

        def make_block_and_layer(i):  # make a dense block followed by a transition layer
            return nn.Sequential(DenseBlock(convolutions[i], self.image_size, dense_out_channels, num_convs),
                                 TransitionLayer(convolutions[i] + dense_out_channels * num_convs,
                                                 convolutions[i+1], ks=ks, st=st, p=p))

        # make dense net. ModuleList required (instead of list) to activate cuda
        self.block_and_layer = nn.ModuleList([make_block_and_layer(i) for i in range(num_convs)])

        # make fully connected layers (reduces loss by ~20%)
        self.bn = nn.BatchNorm1d(num_features=convolutions[-1])
        self.image_outsize = calc_image_out_size(self.image_size, ks, p, st, num_convs)  # image size change due to transitional layers (dense blocks don't change image size)
        fc_layers = [convolutions[-1] * self.image_outsize] + fc_hidden + [output_size]  # include first and last layer

        def make_fully_connected_net(i):  # make a fully connected neural network
            return FullyConnected(fc_layers, i, num_channels)

        self.fully_connected = nn.ModuleList([make_fully_connected_net(i) for i in range(len(fc_layers) - 1)])

    def forward(self, x):
        out = self.relu(self.lowconv(x))  # first convolution

        for block_and_layer in self.block_and_layer:
            out = block_and_layer(out)  # run a dense block followed by transition layer

        out = self.bn(out)
        out = out.view(-1, self.num_channels, self.convolutions[-1] * self.image_outsize)  # resize for num of channels

        for fc_layer in self.fully_connected:
            out = fc_layer(out)  # run a fully connected layer

        # outputs: ClCd, angle
        return out[:, :, 0], out[:, :, 1]
