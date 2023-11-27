import torch.nn as nn


class PixelUnshuffle(nn.Module):

    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        R = self.downscale_factor
        inputs = inputs.reshape(B, C, H // R, R, W // R, R)
        outputs = inputs.permute(0, 1, 3, 5, 2, 4)
        outputs = outputs.reshape(B, C * R ** 2, H // R, W // R)
        return outputs


class PackingBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=8,
        kernel_size=3,
        stride=2,
        padding=1,
        num_groups=16,
    ):
        super().__init__()
        self.pixel_unshuffle = PixelUnshuffle(stride)
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=depth,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels * (stride ** 2) * depth,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),
            nn.ELU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.pixel_unshuffle(inputs)
        outputs = outputs.unsqueeze(1)
        outputs = self.conv3d(outputs)
        outputs = outputs.flatten(1, 2)
        outputs = self.conv2d(outputs)
        return outputs


class UnpackingBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=8,
        kernel_size=3,
        stride=2,
        padding=1,
        num_groups=16,
    ):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * stride ** 2 // depth,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels * stride ** 2 // depth,
            ),
            nn.ELU(inplace=True),
        )
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=depth,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(stride)

    def forward(self, inputs):
        outputs = self.conv2d(inputs)
        outputs = outputs.unsqueeze(1)
        outputs = self.conv3d(outputs)
        outputs = outputs.flatten(1, 2)
        outputs = self.pixel_shuffle(outputs)
        return outputs
