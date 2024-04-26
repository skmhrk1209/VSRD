import collections

import torch.nn as nn


class SqueezeExcitation(nn.Sequential):

    def __init__(self, in_channels, squeeze_channels):
        super().__init__(collections.OrderedDict(
            pool=nn.AdaptiveAvgPool2d(1),
            conv1=nn.Conv2d(
                in_channels=in_channels,
                out_channels=squeeze_channels,
                kernel_size=1,
                bias=True,
            ),
            actv1=nn.ReLU(inplace=True),
            conv2=nn.Conv2d(
                in_channels=squeeze_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=True,
            ),
            actv2=nn.Sigmoid(),
        ))

    def forward(self, inputs):
        return super().forward(inputs) * inputs
