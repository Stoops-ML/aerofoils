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
    def __init__(self, in_channels, image_size):
        super().__init__()
        kernel_size = 3
        stride = 1  # no reduction in image size
        padding = (stride * (image_size - 1) - image_size + kernel_size) // 2  # no reduction in image size

        # channels = [in_channels, 16, 32, 64]  # sum(channels) must equal number of input channels to transition layer

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_features=in_channels)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=in_channels+32, out_channels=32,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=in_channels+64, out_channels=32,
                               kernel_size=kernel_size, stride=stride, padding=padding)

        # first denseblock:
        # in_channels = 64 (output of lowconv)
        # in_channels + 32*3 = 160 because of the concatenations
        # therefore input to transitionlayer1 is 160

    def forward(self, x):
        out = self.bn(x)  # why is batchnorm done here?
        self.out_conv = out  # todo figure out how to pass out_conv to dense() without 'self.'. nonlocal out_conv didn't work

        def dense(conv_func, dense_in):
            conv = self.relu(conv_func(dense_in))
            self.out_conv = torch.cat([self.out_conv, conv], 1)  # concatenate in channel dimension
            dense_out = self.relu(self.out_conv)
            return dense_out

        conv_funcs = [self.conv1, self.conv2, self.conv3]
        for conv in conv_funcs:
            out = dense(conv, out)

        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=0)  # image size reduction

    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)
        return out


class DenseNet(nn.Module):
    def __init__(self, fc_hidden, output_size, dense_channels, transition_channels, image_size):
        super().__init__()

        self.image_size = image_size
        self.num_channels = dense_channels[0]
        self.transitions = transition_channels
        self.image_out_size = transition_channels[-1][-1]

        self.lowconv = nn.Conv1d(in_channels=self.num_channels, out_channels=64,
                                 kernel_size=7, padding=3, bias=False)
        self.relu = nn.ReLU()

        # dense blocks
        self.denseblock1 = nn.Sequential(DenseBlock(64, self.image_size))  # out_channels of lowconv
        self.denseblock2 = nn.Sequential(DenseBlock(46, self.image_size))  # out_channels of transiontlayer1
        self.denseblock3 = nn.Sequential(DenseBlock(46, self.image_size))

        # transition layers
        # self.transitionLayer1 = nn.Sequential(TransitionLayer(*transition_channels[0]))
        self.transitionLayer1 = nn.Sequential(TransitionLayer(160, 46))
        self.transitionLayer2 = nn.Sequential(TransitionLayer(142, 46))
        self.transitionLayer3 = nn.Sequential(TransitionLayer(142, 30))

        # Classifier
        self.bn = nn.BatchNorm1d(num_features=30)  # out_channels of transitionlayer3

        self.fully_connected = nn.Sequential(
            nn.Linear(30 * 24, fc_hidden[0]),
            nn.BatchNorm1d(num_features=self.num_channels),
            nn.LeakyReLU(),

            nn.Linear(fc_hidden[0], output_size),
            nn.BatchNorm1d(num_features=self.num_channels),
        )

    def forward(self, x):
        out = self.relu(self.lowconv(x))

        out = self.denseblock1(out)
        out = self.transitionLayer1(out)

        out = self.denseblock2(out)
        out = self.transitionLayer2(out)

        out = self.denseblock3(out)
        out = self.transitionLayer3(out)

        out = self.bn(out)
        out = out.view(-1, self.num_channels, 30 * 24)  # 30 because output of transitionlayer3
        # todo watch video from python engineer to find out why it's 24 in out.view

        ClCd = out[:, :, 0]
        angle = out[:, :, 1]

        return ClCd, angle
