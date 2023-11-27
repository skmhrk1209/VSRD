import torch
import torch.nn as nn


class DropPath(nn.Module):

    def __init__(self, drop_prob):
        super().__init__()
        keep_prob = 1.0 - drop_prob
        self.bernoulli = torch.distributions.Bernoulli(keep_prob)
        self.drop_prob = drop_prob
        self.keep_prob = keep_prob

    def forward(self, inputs):
        if self.training:
            shape = [*inputs.shape[:1], *[1] * len(inputs.shape[1:])]
            bernoulli = self.bernoulli.sample(shape).to(inputs)
            inputs =  inputs * bernoulli / self.keep_prob
        return inputs
