import torch
import torch.nn as nn


def grid_sampler(inputs, grids, interp_mode="bilinear", padding_mode="zeros", align_corners=True):

    assert interp_mode in ["bilinear"]
    assert padding_mode in ["zeros", "border", "reflection"]
    assert align_corners

    x_grids, y_grids = torch.unbind(grids, dim=-1)
    x_grids = (x_grids + 1.0) / 2.0 * (inputs.shape[-1] - 1)
    y_grids = (y_grids + 1.0) / 2.0 * (inputs.shape[-2] - 1)

    padding_modes = {"zeros": "constant", "border": "replicate", "reflection": "reflect"}
    padding_mode = padding_modes[padding_mode]
    inputs = nn.functional.pad(inputs, (1, 1, 1, 1), padding_mode)

    x_grids = torch.clamp(x_grids + 1, 0, inputs.shape[-1] - 1)
    y_grids = torch.clamp(y_grids + 1, 0, inputs.shape[-2] - 1)

    x_indices_0 = torch.clamp(torch.floor(x_grids).long() + 0, 0, inputs.shape[-1] - 1)
    x_indices_1 = torch.clamp(torch.floor(x_grids).long() + 1, 0, inputs.shape[-1] - 1)
    y_indices_0 = torch.clamp(torch.floor(y_grids).long() + 0, 0, inputs.shape[-2] - 1)
    y_indices_1 = torch.clamp(torch.floor(y_grids).long() + 1, 0, inputs.shape[-2] - 1)

    x_weights = x_grids - x_indices_0
    y_weights = y_grids - y_indices_0

    indices_00 = y_indices_0 * inputs.shape[-1] + x_indices_0
    indices_01 = y_indices_0 * inputs.shape[-1] + x_indices_1
    indices_10 = y_indices_1 * inputs.shape[-1] + x_indices_0
    indices_11 = y_indices_1 * inputs.shape[-1] + x_indices_1

    indices_00 = indices_00.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1)
    indices_01 = indices_01.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1)
    indices_10 = indices_10.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1)
    indices_11 = indices_11.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1)

    samples_00 = torch.gather(inputs.flatten(-2, -1), index=indices_00.flatten(-2, -1), dim=-1).unflatten(-1, indices_00.shape[-2:])
    samples_01 = torch.gather(inputs.flatten(-2, -1), index=indices_01.flatten(-2, -1), dim=-1).unflatten(-1, indices_01.shape[-2:])
    samples_10 = torch.gather(inputs.flatten(-2, -1), index=indices_10.flatten(-2, -1), dim=-1).unflatten(-1, indices_10.shape[-2:])
    samples_11 = torch.gather(inputs.flatten(-2, -1), index=indices_11.flatten(-2, -1), dim=-1).unflatten(-1, indices_11.shape[-2:])

    samples_0 = torch.lerp(samples_00, samples_01, x_weights.unsqueeze(1))
    samples_1 = torch.lerp(samples_10, samples_11, x_weights.unsqueeze(1))
    samples = torch.lerp(samples_0, samples_1, y_weights.unsqueeze(1))

    return samples
