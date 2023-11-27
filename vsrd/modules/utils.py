import torch
import torch.nn as nn


class Residual(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


class Concat(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return torch.cat((inputs, self.module(inputs)), dim=1)


class Lambda(nn.Module):

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class Sequential(nn.Sequential):
    def forward(self, *args, **kwargs):
        for index, module in enumerate(self):
            if index:
                if isinstance(outputs, tuple):
                    outputs = module(*outputs)
                else:
                    outputs = module(outputs)
            else:
                outputs = module(*args, **kwargs)
        return outputs
