import torch
import torch.nn as nn
import numpy as np


class SinusoidalEncoder(nn.Module):

    def __init__(self, num_frequencies):
        super().__init__()
        # NOTE: In practice, I omit the pi in the code to account for the fact that the scene bounds are a bit loose
        # https://github.com/bmild/nerf/issues/12
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_frequencies) * np.pi)

    def forward(self, inputs):
        outputs = torch.stack([
            torch.cos(self.frequencies * inputs.unsqueeze(-1)),
            torch.sin(self.frequencies * inputs.unsqueeze(-1)),
        ], dim=-1).flatten(-3, -1)
        return outputs
