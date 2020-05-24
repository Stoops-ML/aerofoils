import torch.nn as nn
import torch


class ConvNet(nn.Module):
    def __init__(self, input_size, convolutions, num_channels, hidden_layers, output_size):
        super(ConvNet, self).__init__()

        # initialise
        self.num_channels = num_channels
        self.convolutions = convolutions

        # convolutions
        kernel_size = 3
        stride = 1
        padding = 0

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
    def __init__(self, input_size, convolutions):
        super().__init__()
        kernel_size = 3
        stride = 1  # no reduction in image size
        padding = 0  # todo add padding so that image size doesn't reduce

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(num_features=input_size)

        self.conv1 = nn.Conv1d(in_channels=convolutions[0], out_channels=convolutions[1],
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=convolutions[1], out_channels=convolutions[2],
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=convolutions[2], out_channels=convolutions[3],
                               kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.bn(x)  # why is batchnorm done here?
        self.out_conv = out  # todo figure out how to pass out_conv to dense() without 'self.'

        def dense(conv_func, dense_in):
            conv = self.relu(conv_func(dense_in))
            self.out_conv = torch.cat([self.out_conv, conv], 1)  # concatenate in channel dimension
            dense_out = self.relu(self.out_conv)
            return dense_out

        conv_funcs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        for conv in conv_funcs:
            out = dense(conv, out)

        # conv2 = self.relu(self.conv2(conv1))
        # c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        #
        # conv3 = self.relu(self.conv3(c2_dense))
        # c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        #
        # conv4 = self.relu(self.conv4(c3_dense))
        # c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
        #
        # conv5 = self.relu(self.conv5(c4_dense))
        # c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

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
    def __init__(self, num_channels, fc_hidden, output_size, dense_channels, transition_channels):
        super().__init__()

        self.num_channels = num_channels

        self.lowconv = nn.Conv1d(in_channels=num_channels, out_channels=dense_channels[0],
                                 kernel_size=7, padding=3, bias=False)
        self.relu = nn.ReLU()

        # dense blocks
        self.denseblock1 = self.make_layer_or_block(DenseBlock, dense_channels[0])
        self.denseblock2 = self.make_layer_or_block(DenseBlock, dense_channels[1])
        self.denseblock3 = self.make_layer_or_block(DenseBlock, dense_channels[2])
        # self.denseblock1 = self._make_dense_block(DenseBlock, 64)
        # self.denseblock2 = self._make_dense_block(DenseBlock, 128)
        # self.denseblock3 = self._make_dense_block(DenseBlock, 128)

        # transition layers
        self.transitionLayer1 = self.make_layer_or_block(TransitionLayer, transition_channels[0])
        self.transitionLayer2 = self.make_layer_or_block(TransitionLayer, transition_channels[1])
        self.transitionLayer3 = self.make_layer_or_block(TransitionLayer, transition_channels[2])
        # self.transitionLayer1 = self._make_transition_layer(TransitionLayer, in_channels=160, out_channels=128)
        # self.transitionLayer2 = self._make_transition_layer(TransitionLayer, in_channels=160, out_channels=128)
        # self.transitionLayer3 = self._make_transition_layer(TransitionLayer, in_channels=160, out_channels=64)

        # Classifier
        # self.bn = nn.BatchNorm1d(num_features=self.tl_out)
        # self.pre_classifier = nn.Linear(self.tl_out * 4 * 4, 512)
        # self.classifier = nn.Linear(512, nr_classes)

        self.fully_connected = nn.Sequential(
            nn.Linear(self.image_size * transition_channels[2], fc_hidden[0]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.LeakyReLU(),

            nn.Linear(fc_hidden[0], output_size),
            nn.BatchNorm1d(num_features=num_channels),
        )

    # def _make_dense_block(self, block, in_channels):
    #     layers = []
    #     layers.append(block(in_channels))
    #     return nn.Sequential(*layers)

    # def _make_transition_layer(self, layer, in_channels, out_channels):
    #     modules = []
    #     modules.append(layer(in_channels, out_channels))
    #     return nn.Sequential(*modules)

    def make_layer_or_block(self, layer_or_block, *args):
        if len(args) == 1:
            return nn.Sequential(*layer_or_block(args))  # dense block: args = in_channels
        elif len(args) == 2:
            return nn.Sequential(*layer_or_block(args[0], args[1]))  # transition layer: args = in_channels,out_channels
        else:
            raise ValueError('A dense block requires number of input channels. Alternatively a transition layer '
                             'requires number of input channels and number of output channels.')

    def forward(self, x):
        out = self.relu(self.lowconv(x))

        out = self.denseblock1(out)
        out = self.transitionLayer1(out)

        out = self.denseblock2(out)
        out = self.transitionLayer2(out)

        out = self.denseblock3(out)
        out = self.transitionLayer3(out)

        out = self.bn(out)
        out = out.view(-1, self.num_channels, self.tl_out * 4 * 4)

        # out = self.pre_classifier(out)
        # out = self.classifier(out)

        out = self.fully_connected(out)
        ClCd = out[:, :, 0]
        angle = out[:, :, 1]

        return ClCd, angle
