import torch.nn as nn
import torch
import sys


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
        super().__init__()
        kernel_size = 3
        stride = 1  # no reduction in image size
        padding = (stride * (image_size - 1) - image_size + kernel_size) // 2  # no reduction in image size
        self.num_convs = num_convs

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_features=in_channels)

        def make_conv(i):
            return nn.Sequential(nn.Conv1d(in_channels=in_channels + dense_out_channels*i,
                                           out_channels=dense_out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding))

        self.convs = [make_conv(i) for i in range(num_convs)]

    def forward(self, x):
        out = self.bn(x)  # why is batchnorm done here?
        self.out_conv = out  # todo figure out how to pass out_conv to dense() without 'self.'. nonlocal out_conv didn't work

        def dense(conv_func, dense_in):
            conv = self.relu(conv_func(dense_in))
            self.out_conv = torch.cat([self.out_conv, conv], 1)  # concatenate in channel dimension
            dense_out = self.relu(self.out_conv)
            return dense_out

        for conv in self.convs:
            out = dense(conv, out)

        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, st=2, p=0):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool1d(kernel_size=ks, stride=st, padding=p)  # image size reduction

    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)
        return out


class DenseNet(nn.Module):
    def __init__(self, image_size, output_size, convolutions, fc_hidden, num_channels, dense_out_channels):
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
                                 padding=(1 * (image_size - 1) - image_size + 7) // 2,  # image size unchanged
                                 bias=False)  # TODO: bias terms set to false. Set them to true?

        def make_block_and_layer(i):
            return nn.Sequential(DenseBlock(convolutions[i], self.image_size, dense_out_channels, num_convs),
                                 TransitionLayer(convolutions[i] + dense_out_channels * num_convs,
                                                 convolutions[i+1], ks=ks, st=st, p=p))

        # make dense block followed by transition layer
        self.block_and_layer = [make_block_and_layer(i) for i in range(num_convs)]

        # image size change due to transitional layers (dense blocks don't change image size)
        self.image_outsize = self.image_size  # initialise
        for _ in range(num_convs):
            self.image_outsize = (self.image_outsize - ks + 2 * p) // st + 1

        # fully connected layer (reduces loss by ~20%)
        self.bn = nn.BatchNorm1d(num_features=convolutions[-1])
        self.fully_connected = nn.Sequential(
            nn.Linear(convolutions[-1] * self.image_outsize, fc_hidden[0]),
            nn.BatchNorm1d(num_features=self.num_channels),
            nn.LeakyReLU(),

            nn.Linear(fc_hidden[0], output_size),
            nn.BatchNorm1d(num_features=self.num_channels),
        )

    def forward(self, x):
        out = self.relu(self.lowconv(x))

        for block_and_layer in self.block_and_layer:
            out = block_and_layer(out)

        out = self.bn(out)
        out = out.view(-1, self.num_channels, self.convolutions[-1] * self.image_outsize)
        out = self.fully_connected(out)

        ClCd = out[:, :, 0]
        angle = out[:, :, 1]

        return ClCd, angle
