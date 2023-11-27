import torch
import torch.nn as nn


class Scale(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(scale))

    def forward(self, inputs):
        return self.weight * inputs


class LayerScale(nn.Module):

    def __init__(self, num_channels, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_channels, 1, 1), epsilon))
        self.num_channels = num_channels

    def forward(self, inputs):
        return self.weight * inputs
