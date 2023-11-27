import torch
import torch.nn as nn


class SinkhornKnopp(nn.Module):
    """ Sinkhorn Knopp Iteration

    References:
        - [OTA: Optimal Transport Assignment for Object Detection](https://arxiv.org/abs/2103.14259)
    """

    def __init__(self, gamma=0.1, num_steps=50, epsilon=1e-6):
        super().__init__()
        self.gamma = gamma
        self.num_steps = num_steps
        self.epsilon = epsilon

    def forward(self, C, d, s):
        d = torch.log(d + self.epsilon)
        s = torch.log(s + self.epsilon)
        u = torch.ones_like(d)
        v = torch.ones_like(s)
        M = -C / self.gamma
        for _ in range(self.num_steps):
            u = d - torch.logsumexp(M + v[None, ...], dim=1)
            v = s - torch.logsumexp(M + u[..., None], dim=0)
        P = u[..., None] + M + v[None, ...]
        return torch.exp(P)
