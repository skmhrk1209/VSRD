import torch
import torch.nn as nn
import numpy as np

from .. import utils


@utils.tensor_args(dtype=torch.long, device="cuda")
def conv_output_size(input_size, kernel_size, stride, padding, dilation):
    return (input_size + padding * 2 - dilation * (kernel_size - 1) - 1) // stride + 1


class MultiHeadLocalAttention2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        num_heads=1,
        bias=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.num_heads = num_heads

        self.query_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.key_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.value_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.output_projector = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

        self.query_unfolder = nn.Unfold(
            kernel_size=1,
            stride=stride,
        )
        self.key_unfolder = nn.Unfold(
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )
        self.value_unfolder = nn.Unfold(
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, queries, keys, values):

        input_size = values.shape[-2:]

        # ----------------------------------------------------------------
        # query

        queries = self.query_projector(queries)
        # -> [B, C, H, W]
        queries = self.query_unfolder(queries)
        # -> [B, C * 1 * 1, N]
        queries = queries.unflatten(1, (self.num_heads, self.out_channels // self.num_heads, 1))
        # -> [B, M, C // M, 1 * 1, N]
        queries = queries.permute(0, 4, 1, 2, 3)
        # -> [B, N, M, C // M, 1 * 1]

        # ----------------------------------------------------------------
        # key

        keys = self.key_projector(keys)
        # -> [B, C, H, W]
        keys = self.key_unfolder(keys)
        # -> [B, C * KH * KW, N]
        keys = keys.unflatten(1, (self.num_heads, self.out_channels // self.num_heads, np.prod(np.broadcast_to(self.kernel_size, [2]))))
        # -> [B, M, C // M, KH * KW, N]
        keys = keys.permute(0, 4, 1, 2, 3)
        # -> [B, N, M, C // M, KH * KW]

        # ----------------------------------------------------------------
        # value

        values = self.value_projector(values)
        # -> [B, C, H, W]
        values = self.value_unfolder(values)
        # -> [B, C * KH * KW, N]
        values = values.unflatten(1, (self.num_heads, self.out_channels // self.num_heads, np.prod(np.broadcast_to(self.kernel_size, [2]))))
        # -> [B, M, C // M, KH * KW, N]
        values = values.permute(0, 4, 1, 2, 3)
        # -> [B, N, M, C // M, KH * KW]

        # ----------------------------------------------------------------
        # attention

        attentions = queries.transpose(-2, -1) @ keys
        # -> [B, N, M, 1 * 1, KH * KW]
        attentions = nn.functional.softmax(attentions / np.sqrt(self.out_channels), dim=-1)
        # -> [B, N, M, 1 * 1, KH * KW]

        # ----------------------------------------------------------------
        # aggregation

        outputs = values @ attentions.transpose(-2, -1)
        # -> [B, N, M, C // M, 1 * 1]
        outputs = outputs.permute(0, 2, 3, 4, 1)
        # -> [B, M, C // M, 1 * 1, N]
        output_size = conv_output_size(input_size, self.kernel_size, self.stride, self.padding, self.dilation)
        outputs = outputs.flatten(1, 3).unflatten(-1, output_size)
        # -> [B, C, H', W']

        # ----------------------------------------------------------------
        # output

        outputs = self.output_projector(outputs)
        # -> [B, C, H', W']

        return outputs


class MultiHeadGlobalAttention2d(nn.Module):

    def __init__(self, in_channels, out_channels, num_heads=1, bias=True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.query_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.key_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.value_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.out_projector = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, queries, keys, values):

        input_size = values.shape[-2:]

        # ----------------------------------------------------------------
        # query

        queries = self.query_projector(queries)
        # -> [B, C, H, W]
        queries = queries.flatten(-2, -1).unflatten(1, (self.num_heads, self.out_channels // self.num_heads))
        # -> [B, M, C // M, H * W]

        # ----------------------------------------------------------------
        # key

        keys = self.key_projector(keys)
        # -> [B, C, H, W]
        keys = keys.flatten(-2, -1).unflatten(1, (self.num_heads, self.out_channels // self.num_heads))
        # -> [B, M, C // M, H * W]

        # ----------------------------------------------------------------
        # value

        values = self.value_projector(values)
        # -> [B, C, H, W]
        values = values.flatten(-2, -1).unflatten(1, (self.num_heads, self.out_channels // self.num_heads))
        # -> [B, M, C // M, H * W]

        # ----------------------------------------------------------------
        # attention

        attentions = queries.transpose(-2, -1) @ keys
        # -> [B, M, H * W, H * W]
        attentions = nn.functional.softmax(attentions / np.sqrt(self.out_channels), dim=-1)
        # -> [B, M, H * W, H * W]

        # ----------------------------------------------------------------
        # aggregation

        outputs = values @ attentions.transpose(-2, -1)
        # -> [B, M, C // M, H * W]
        outputs = outputs.flatten(1, 2).unflatten(-1, input_size)
        # -> [B, C, H, W]

        # ----------------------------------------------------------------
        # output

        outputs = self.output_projector(outputs)
        # -> [B, C, H, W]

        return outputs


class MultiHeadDeformableAttention(nn.Module):

    def __init__(self, in_channels, out_channels, num_samples, num_heads=1, bias=True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_samples = num_samples
        self.num_heads = num_heads

        self.location_predictor = nn.Linear(
            in_features=in_channels,
            out_features=num_samples * 2,
            bias=bias,
        )
        self.query_projector = nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
            bias=bias,
        )
        self.key_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.value_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.output_projector = nn.Linear(
            in_features=out_channels,
            out_features=out_channels,
            bias=bias,
        )

    def forward(self, queries, keys, values):

        # ----------------------------------------------------------------
        # location

        locations = self.location_predictor(queries)
        locations = torch.tanh(locations)
        # -> [B, N, S * 2]
        locations = locations.unflatten(-1, (self.num_samples, 2))
        # -> [B, N, S, 2]

        # ----------------------------------------------------------------
        # query

        queries = self.query_projector(queries)
        # -> [B, N, C]
        queries = queries.unsqueeze(-1)
        # -> [B, N, C, 1]
        queries = queries.unflatten(-2, (self.num_heads, self.out_channels // self.num_heads))
        # -> [B, N, M, C // M, 1]

        # ----------------------------------------------------------------
        # key

        keys = self.key_projector(keys)
        # -> [B, C, H, W]
        keys = nn.functional.grid_sample(
            input=keys,
            grid=locations,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # -> [B, C, N, S]
        keys = keys.permute(0, 2, 1, 3)
        # -> [B, N, C, S]
        keys = keys.unflatten(-2, (self.num_heads, self.out_channels // self.num_heads))
        # -> [B, N, M, C // M, S]

        # ----------------------------------------------------------------
        # value

        values = self.value_projector(values)
        # -> [B, C, H, W]
        values = nn.functional.grid_sample(
            input=values,
            grid=locations,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # -> [B, C, N, S]
        values = values.permute(0, 2, 1, 3)
        # -> [B, N, C, S]
        values = values.unflatten(-2, (self.num_heads, self.out_channels // self.num_heads))
        # -> [B, N, M, C // M, S]

        # ----------------------------------------------------------------
        # attention

        attentions = queries.transpose(-2, -1) @ keys
        # -> [B, N, M, 1, S]
        attentions = nn.functional.softmax(attentions / np.sqrt(self.out_channels), dim=-1)
        # -> [B, N, M, 1, S]

        # ----------------------------------------------------------------
        # aggregation

        outputs = values @ attentions.transpose(-2, -1)
        # -> [B, N, M, C // M, 1]
        outputs = outputs.flatten(-3, -1)
        # -> [B, N, C]

        # ----------------------------------------------------------------
        # output

        outputs = self.output_projector(outputs)
        # -> [B, N, C]

        return outputs
