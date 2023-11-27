import torch
import torch.nn as nn

from .. import utils


class SpatialPropagation2d(nn.Module):
    """ Anisotropic Diffusion Process
    References:
        - [Learning Affinity via Spatial Propagation Networks](https://arxiv.org/abs/1710.01020)
        - [Learning Depth with Convolutional Spatial Propagation Network](https://arxiv.org/abs/1810.02695)
    """

    def __init__(self, kernel_size, padding, stride, num_steps=10, epsilon=1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.unfolder = nn.Unfold(
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def forward(self, inputs, kernels):
        # -> [B, C, KH, KW, H, W]
        y = torch.arange(kernels.shape[2], device=kernels.device)
        x = torch.arange(kernels.shape[3], device=kernels.device)
        y, x = torch.meshgrid(y, x, indexing="ij")
        masks = ~((y == (kernels.shape[2] - 1) // 2) & (x == (kernels.shape[3] - 1) // 2))
        # -> [1, 1, KH, KW, 1, 1]
        neighbors = kernels * utils.unsqueeze_as(masks, kernels, 2)
        norm_factor = torch.sum(torch.abs(neighbors), dim=(2, 3), keepdim=True)
        neighbors = neighbors / (norm_factor + self.epsilon)
        # -> [B, C, KH, KW, H, W]
        centers = 1.0 - torch.sum(neighbors, dim=(2, 3))
        # -> [B, C, H, W]
        outputs = inputs
        for _ in range(self.num_steps):
            outputs = self.unfolder(outputs)
            # -> [B, C * KH * KW, N]
            outputs = outputs.reshape(*inputs.shape[:2], self.kernel_size, self.kernel_size, *inputs.shape[-2:])
            # -> [B, C, KH, KW, H, W]
            outputs = inputs * centers + torch.sum(outputs * neighbors, dim=(2, 3))
            # -> [B, C, H, W]
        return outputs
