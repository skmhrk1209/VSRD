import torch
import torch.nn as nn


class GradScale(nn.Module):

    class GradScaleFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, inputs, scale):
            ctx.scale = scale
            return inputs

        @staticmethod
        def backward(ctx, output_gradients):
            return output_gradients * ctx.scale, None

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, inputs):
        return __class__.GradScaleFunction.apply(inputs, self.scale)
