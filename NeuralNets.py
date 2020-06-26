import torch.nn as nn
import torch
import sys


def padding_no_size_reduction(image_size, stride, kernel_size):
    """returns padding for convolution such that input image size = output image size"""
    return (stride * (image_size - 1) - image_size + kernel_size) // 2


def image_out_size(image_in_size, kernel_size, padding, stride, num_layers):
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
    def __init__(self, in_channels, dense_out_channels, num_convs, ks, st, p):
        """make a dense block"""
        super().__init__()
        # self.num_convs = num_convs  # todo delete this after interview

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_features=in_channels)

        def conv_layer(i):  # make a layer in the dense block
            return nn.Conv1d(in_channels=in_channels + dense_out_channels * i,
                             out_channels=dense_out_channels, kernel_size=ks, stride=st, padding=p)

        self.dense_block = nn.ModuleList([conv_layer(i) for i in range(num_convs)])  # list of layers in dense block

    def forward(self, x):
        out = self.bn(x)

        def dense(conv_func, dense_in):
            # nonlocal out  # todo I don't think this is needed as it's a function
            conv = self.relu(conv_func(dense_in))  # do convolution
            out_conv = torch.cat([out, conv], 1)  # concatenate in channel dimension
            dense_out = self.relu(out_conv)
            return dense_out

        for conv in self.dense_block:
            out = dense(conv, out)

        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ks, st, p):
        """make a transition layer"""
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        # todo ks, st, p variables affect avg_pool NOT CONVOLUTION! Change this
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)  # image size reduction
        self.avg_pool = nn.AvgPool1d(kernel_size=ks, stride=st, padding=p)  # image size reduction

    def forward(self, x):
        return self.avg_pool(self.bn(self.relu(self.conv(x))))  # todo FC and old convnet did linear, then bn, then relu


class FullyConnected(nn.Module):
    def __init__(self, channels_in, channels_out, num_channels):
        """make layer in fully connected neural network"""
        super().__init__()

        self.linear = nn.Linear(channels_in, channels_out)
        self.bn = nn.BatchNorm1d(num_features=num_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.linear(x)))


class DenseNet(nn.Module):
    def __init__(self, image_size, output_size, convolutions, hidden_channels, num_channels, dense_out_channels):
        """create a densely connected convolutional neural network"""
        super().__init__()

        # low convolution
        self.lowconv = nn.Conv1d(in_channels=num_channels, out_channels=convolutions[0], stride=1, kernel_size=7,
                                 padding=padding_no_size_reduction(image_size, 1, 7),  # image size unchanged
                                 bias=False)  # TODO: bias terms set to false. Set them to true?
        self.relu = nn.ReLU()

        # dense block and transition layer values
        dense_ks, dense_st = 3, 1
        trans_ks, trans_pad, trans_st = 3, 0, 2

        def make_block_and_layer(i):  # make a dense block followed by a transition layer
            dense_pad = padding_no_size_reduction(self.image_outsize, dense_st, dense_ks)  # dense block: image size in = size out
            self.image_outsize = image_out_size(self.image_outsize, trans_ks, trans_pad, trans_st, 1)  # iamge size reduction due to transition layer
            return nn.Sequential(DenseBlock(convolutions[i], dense_out_channels, num_convs,
                                            dense_ks, dense_st, dense_pad),
                                 TransitionLayer(convolutions[i] + dense_out_channels * num_convs,
                                                 convolutions[i+1], trans_ks, trans_st, trans_pad))

        # make dense net. ModuleList required (instead of list) to activate cuda
        self.convolutions = convolutions
        self.image_outsize = image_size
        num_convs = len(convolutions) - 1  # num convs in dense block (-1 for lowconv, included in list 'convolutions')
        self.block_and_layer = nn.ModuleList([make_block_and_layer(i) for i in range(num_convs)])

        # fully connected layers
        self.num_channels = num_channels  # number of input channels
        self.bn = nn.BatchNorm1d(num_features=convolutions[-1])
        num_hidden = len(hidden_channels)  # num hidden layers (see 'fc_layers')
        fc_channels = [convolutions[-1] * self.image_outsize] + hidden_channels  # channels of NN

        def make_fully_connected_net(i):  # make a fully connected neural network
            return FullyConnected(fc_channels[i], fc_channels[i+1], num_channels)

        self.fully_connected = nn.ModuleList([make_fully_connected_net(i) for i in range(num_hidden)])

        self.final_layer = nn.Sequential(nn.Linear(hidden_channels[-1], output_size),
                                         nn.BatchNorm1d(num_channels))  # no activation function (no relu)

    def forward(self, x):
        out = self.relu(self.lowconv(x))  # first convolution

        for block_and_layer in self.block_and_layer:
            out = block_and_layer(out)  # run a dense block followed by transition layer

        out = self.bn(out)  # TODO transition layer outputs batchnorm followed by average pooling -> need another bn?
        out = out.view(-1, self.num_channels, self.convolutions[-1] * self.image_outsize)  # resize for num of channels

        for fc_layer in self.fully_connected:
            out = fc_layer(out)  # run a fully connected layer

        out = self.final_layer(out)

        # outputs: ClCd, angle
        return out[:, :, 0], out[:, :, 1]
