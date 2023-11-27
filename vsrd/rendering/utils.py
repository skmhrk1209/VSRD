import torch
import torch.nn as nn


def ray_casting(image_size, intrinsic_matrices, extrinsic_matrices):
    cartesian_grid = torch.stack(list(reversed(torch.meshgrid(*map(torch.arange, image_size), indexing="ij"))), dim=-1)
    cartesian_grid = nn.functional.pad(cartesian_grid, (0, 1), mode="constant", value=1.0)
    inverse_intrinsic_matrices = torch.linalg.inv(intrinsic_matrices)
    inverse_extrinsic_matrices = torch.linalg.inv(extrinsic_matrices)
    inverse_projection_matrices = inverse_extrinsic_matrices[..., :3, :3] @ inverse_intrinsic_matrices
    ray_directions = torch.einsum(
        "...mn,hwn->...hwm",
        inverse_projection_matrices,
        cartesian_grid.to(inverse_projection_matrices),
    )
    ray_directions = nn.functional.normalize(ray_directions, dim=-1)
    camera_positions = inverse_extrinsic_matrices[..., :3, 3]
    return camera_positions, ray_directions
