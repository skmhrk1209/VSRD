import torch
import torch.nn as nn
import numpy as np

from ... import modules


class TensorialCPEncoder(nn.Module):

    def __init__(self, grid_resolution, num_components):
        super().__init__()
        self.vectors = nn.ParameterList([
            nn.Parameter(torch.randn(num_components, resolution))
            for resolution in grid_resolution
        ])

    def forward(self, positions):

        features = torch.prod(torch.stack([
            modules.grid_sampler(
                inputs=vectors.unsqueeze(-2).unsqueeze(0),
                grids=torch.stack([
                    vector_positions,
                    torch.zeros_like(vector_positions),
                ], dim=-1).flatten(0, -2).unsqueeze(-3).unsqueeze(0),
                interp_mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(-2).squeeze(0).t().unflatten(0, positions.shape[:-1])
            for vector_positions, vectors
            in zip(torch.unbind(positions, dim=-1), self.vectors)
        ], dim=0), dim=0)

        return features


class TensorialVMEncoder(TensorialCPEncoder):

    def __init__(self, grid_resolution, num_components):
        super().__init__(grid_resolution, num_components)
        self.matrices = nn.ParameterList([
            nn.Parameter(torch.randn(num_components, *resolution))
            for resolution in zip(np.roll(grid_resolution, -1), np.roll(grid_resolution, -2))
        ])

    def forward(self, positions):

        features = torch.cat([
            torch.mul(
                modules.grid_sampler(
                    inputs=vectors.unsqueeze(-2).unsqueeze(0),
                    grids=torch.stack([
                        vector_positions,
                        torch.zeros_like(vector_positions),
                    ], dim=-1).flatten(0, -2).unsqueeze(-3).unsqueeze(0),
                    interp_mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).squeeze(-2).squeeze(0).t().unflatten(0, positions.shape[:-1]),
                modules.grid_sampler(
                    inputs=matrices.unsqueeze(0),
                    grids=(
                        torch.stack(matrix_positions, dim=-1)
                        .flatten(0, -2).unsqueeze(-3).unsqueeze(0)
                    ),
                    interp_mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).squeeze(-2).squeeze(0).t().unflatten(0, positions.shape[:-1]),
            )
            for vector_positions, *matrix_positions, vectors, matrices
            in zip(
                torch.unbind(torch.roll(positions, -1, dims=-0), dim=-1),
                torch.unbind(torch.roll(positions, -1, dims=-1), dim=-1),
                torch.unbind(torch.roll(positions, -1, dims=-2), dim=-1),
                self.vectors,
                self.matrices,
            )
        ], dim=-1)

        return features
