import operator

import torch
import torch.nn as nn


class HyperRadianceField(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels_list,
        hyper_in_channels,
        hyper_out_channels_list,
    ):
        super().__init__()

        in_channels_list = [in_channels, *out_channels_list]
        out_channels_list = [*out_channels_list, 3]

        num_neurons_list = list(map(
            operator.add,
            out_channels_list,
            map(operator.mul, in_channels_list, out_channels_list),
        ))

        hyper_in_channels_list = [hyper_in_channels, *hyper_out_channels_list]
        hyper_out_channels_list = [*hyper_out_channels_list, sum(num_neurons_list)]

        self.hypernetwork = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.LayerNorm(out_channels),
                    nn.GELU(),
                )
                for in_channels, out_channels
                in zip(hyper_in_channels_list[:-1], hyper_out_channels_list[:-1])
            ],
            *[
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                )
                for in_channels, out_channels
                in zip(hyper_in_channels_list[-1:], hyper_out_channels_list[-1:])
            ],
        )

        self.num_neurons_list = num_neurons_list
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list

        # weight normalization
        # [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)
        self.apply(lambda module: nn.utils.weight_norm(module) if isinstance(module, nn.Linear) else module)

    def radiance_field(self, weights, positions):
        features = positions
        for layer_index, (weights, in_channels, out_channels) in enumerate(zip(
            torch.split(weights, self.num_neurons_list, dim=-1),
            self.in_channels_list,
            self.out_channels_list,
        )):
            if layer_index:
                features = nn.functional.layer_norm(features, [in_channels])
                features = nn.functional.gelu(features)
            features = torch.einsum(
                "...mn,...n->...m",
                weights.unflatten(-1, (out_channels, in_channels + 1)),
                nn.functional.pad(features, (0, 1), mode="constant", value=1.0),
            )
        radiances = features
        return radiances

    def forward(self, embeddings):
        weights = self.hypernetwork(embeddings)
        return weights
