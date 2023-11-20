# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import operator
import functools

import torch
import torch.nn as nn

from .. import utils


def expand_to_4x4(matrices):
    matrices_4x4 = torch.eye(4).to(matrices)
    matrices_4x4 = matrices_4x4.reshape(*[1] * len(matrices.shape[:-2]), 4, 4)
    matrices_4x4 = matrices_4x4.repeat(*matrices.shape[:-2], 1, 1)
    matrices_4x4[..., :matrices.shape[-2], :matrices.shape[-1]] = matrices
    return matrices_4x4


def skew_symmetric_matrix(vectors):
    x, y, z = torch.unbind(vectors, dim=-1)
    zero = torch.zeros_like(x)
    cross_matrices = torch.stack([
        torch.stack([zero,   -z,    y], dim=-1),
        torch.stack([   z, zero,   -x], dim=-1),
        torch.stack([  -y,    x, zero], dim=-1),
    ], dim=-2)
    return cross_matrices


def rotation_matrix_x(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ one, zero,  zero], dim=-1),
        torch.stack([zero,  cos,  -sin], dim=-1),
        torch.stack([zero,  sin,   cos], dim=-1),
    ], dim=-2)
    return rotation_matrices


def rotation_matrix_y(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ cos, zero,  sin], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([-sin, zero,  cos], dim=-1),
    ], dim=-2)
    return rotation_matrices


def rotation_matrix_z(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ cos, -sin, zero], dim=-1),
        torch.stack([ sin,  cos, zero], dim=-1),
        torch.stack([zero, zero,  one], dim=-1),
    ], dim=-2)
    return rotation_matrices


def rotation_matrix(rotation_axes, rotation_angles):
    cos = torch.cos(rotation_angles).unsqueeze(-1).unsqueeze(-2)
    sin = torch.sin(rotation_angles).unsqueeze(-1).unsqueeze(-2)
    rotation_matrices = (
        (1.0 - cos) * torch.einsum("...m,...n->...mn", rotation_axes, rotation_axes) +
        sin * skew_symmetric_matrix(rotation_axes) +
        cos * torch.eye(3).to(rotation_axes)
    )
    return rotation_matrices


def translation_matrix(translation_vectors):
    translation_matrices = torch.eye(4).to(translation_vectors)
    translation_matrices = translation_matrices.reshape(*[1] * len(translation_vectors.shape[:-1]), 4, 4)
    translation_matrices = translation_matrices.repeat(*translation_vectors.shape[:-1], 1, 1)
    translation_matrices[..., :3, 3] = translation_vectors
    return translation_matrices


def essential_matrix(rotation_matrices, translation_vectors):
    essential_matrices = skew_symmetric_matrix(translation_vectors) @ rotation_matrices
    return essential_matrices


def fundamental_matrix(essential_matrices, intrinsic_matrices_1, intrinsic_matrices_2):
    fundamental_matrces = torch.linalg.inv(intrinsic_matrices_2).transpose(-2, -1) @ essential_matrices @ torch.linalg.inv(intrinsic_matrices_1)
    return fundamental_matrces


def projection(coord_maps, intrinsic_matrices, extrinsic_matrices=None):
    # vector to matrix
    coord_maps = coord_maps.unsqueeze(-1)
    if extrinsic_matrices is not None:
        # broadcast to the image plane
        if extrinsic_matrices.ndim == 3:
            extrinsic_matrices = extrinsic_matrices.unsqueeze(-3).unsqueeze(-4)
        # (x, y, z, w) <- (R | t) @ (x, y, z, w)
        coord_maps = extrinsic_matrices @ coord_maps
    # (x, y, z) <- (x / w, y / w, z / w)
    coord_maps = coord_maps[..., :-1, :] / coord_maps[..., -1:, :]
    # broadcast to the image plane
    if intrinsic_matrices.ndim == 3:
        intrinsic_matrices = intrinsic_matrices.unsqueeze(-3).unsqueeze(-4)
    # (x, y, z) <- K @ (x, y, z)
    coord_maps = intrinsic_matrices @ coord_maps
    # matrix to vector
    coord_maps = coord_maps.squeeze(-1)
    return coord_maps


def backprojection(depth_maps, intrinsic_matrices, extrinsic_matrices=None):
    coords_y = torch.arange(depth_maps.shape[-2]).to(depth_maps)
    coords_x = torch.arange(depth_maps.shape[-1]).to(depth_maps)
    coord_maps_y, coord_maps_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
    coord_maps = torch.stack([coord_maps_x, coord_maps_y], dim=-1)
    # homogeneous coordinates
    coord_maps = nn.functional.pad(coord_maps, (0, 1), mode="constant", value=1.0)
    # (x, y, z) <- d × (x, y, 1)
    coord_maps = coord_maps * depth_maps.permute(0, 2, 3, 1)
    # vector to matrix
    coord_maps = coord_maps.unsqueeze(-1)
    # broadcast to the image plane
    if intrinsic_matrices.ndim == 3:
        intrinsic_matrices = intrinsic_matrices.unsqueeze(-3).unsqueeze(-4)
    # (x, y, z) <- inv(K) @ (x, y, z)
    coord_maps = torch.linalg.inv(intrinsic_matrices) @ coord_maps
    # (x, y, z, w) <- (x, y, z, 1)
    coord_maps = nn.functional.pad(coord_maps, (0, 0, 0, 1), mode="constant", value=1.0)
    if extrinsic_matrices is not None:
        # broadcast to the image plane
        if extrinsic_matrices.ndim == 3:
            extrinsic_matrices = extrinsic_matrices.unsqueeze(-3).unsqueeze(-4)
        # (x, y, z, w) <- inv(R | t) @ (x, y, z, w)
        coord_maps = torch.linalg.inv(extrinsic_matrices) @ coord_maps
    # matrix to vector
    coord_maps = coord_maps.squeeze(-1)
    return coord_maps


def backward_warping(
    source_feature_maps,
    target_depth_maps,
    source_intrinsic_matrices,
    target_intrinsic_matrices,
    source_extrinsic_matrices=None,
    target_extrinsic_matrices=None,
    interp_mode="bilinear",
    epsilon=1e-6,
    **kwargs,
):
    target_coord_maps = backprojection(target_depth_maps, target_intrinsic_matrices, target_extrinsic_matrices)
    source_coord_maps = projection(target_coord_maps, source_intrinsic_matrices, source_extrinsic_matrices)
    source_coord_maps_x, source_coord_maps_y, source_coord_maps_z = torch.unbind(source_coord_maps, dim=-1)
    source_coord_maps_x = utils.linear_map(source_coord_maps_x / (source_coord_maps_z + epsilon), 0, source_feature_maps.shape[-1] - 1, -1.0, 1.0)
    source_coord_maps_y = utils.linear_map(source_coord_maps_y / (source_coord_maps_z + epsilon), 0, source_feature_maps.shape[-2] - 1, -1.0, 1.0)
    source_coord_maps = torch.stack([source_coord_maps_x, source_coord_maps_y], dim=-1)
    target_feature_maps = nn.functional.grid_sample(source_feature_maps, source_coord_maps, mode=interp_mode, **kwargs)
    # ----------------------------------------------------------------
    # NOTE: invisible objects
    # ----------------------------------------------------------------
    # objects which are invisible from the source camera should not be projected?
    # the depth estimator is being optimized and the predicted depth map is not accurate now
    # so, they should be projected to facilitate the optimization process
    # even if the objects are actually invisible from the source camera
    # because we cannot know the ground truth depths in a self-supervised manner
    # ----------------------------------------------------------------
    # valid_masks = (target_depth_maps > 0.0) & (source_coord_maps_z.unsqueeze(1) > 0.0)
    # target_feature_maps = torch.where(valid_masks, target_feature_maps, torch.zeros_like(target_feature_maps))
    # ----------------------------------------------------------------
    return target_feature_maps


def forward_warping(
    source_feature_maps,
    source_depth_maps,
    source_intrinsic_matrices,
    target_intrinsic_matrices,
    source_extrinsic_matrices=None,
    target_extrinsic_matrices=None,
    interp_mode="bilinear",
    epsilon=1e-6,
    **kwargs,
):
    source_coord_maps = backprojection(source_depth_maps, source_intrinsic_matrices, source_extrinsic_matrices)
    target_coord_maps = projection(source_coord_maps, target_intrinsic_matrices, target_extrinsic_matrices)
    target_coord_maps_x, target_coord_maps_y, target_coord_maps_z = torch.unbind(target_coord_maps, dim=-1)
    target_coord_maps_x = target_coord_maps_x / (target_coord_maps_z + epsilon)
    target_coord_maps_y = target_coord_maps_y / (target_coord_maps_z + epsilon)
    target_coord_maps = torch.stack([target_coord_maps_x, target_coord_maps_y], dim=-1)
    target_feature_maps = grid_splatting(source_feature_maps, target_coord_maps, interp_mode=interp_mode, **kwargs)
    # ----------------------------------------------------------------
    # NOTE: invisible objects
    # ----------------------------------------------------------------
    # objects which are invisible from the source camera should not be projected?
    # the depth estimator is being optimized and the predicted depth map is not accurate now
    # so, they should be projected to facilitate the optimization process
    # even if the objects are actually invisible from the source camera
    # because we cannot know the ground truth depths in a self-supervised manner
    # ----------------------------------------------------------------
    # valid_masks = (target_depth_maps > 0.0) & (source_coord_maps_z.unsqueeze(1) > 0.0)
    # target_feature_maps = torch.where(valid_masks, target_feature_maps, torch.zeros_like(target_feature_maps))
    # ----------------------------------------------------------------
    return target_feature_maps


def grid_splatting(
    inputs,
    coords,
    image_size=None,
    valid_masks=None,
    interp_mode="bilinear",
    padding_mode="zeros",
):
    assert inputs.ndim == 4
    assert coords.ndim == 4
    assert interp_mode in ("nearest", "bilinear")
    assert padding_mode in ("zeros",)

    batch_coords = torch.arange(inputs.shape[0], device=inputs.device)
    channel_coords = torch.arange(inputs.shape[1], device=inputs.device)

    batch_coords = utils.unsqueeze_as(batch_coords, inputs, start_dim=0).expand_as(inputs)
    channel_coords = utils.unsqueeze_as(channel_coords, inputs, start_dim=1).expand_as(inputs)

    spatial_coords = coords.unsqueeze(1).expand(*inputs.shape, -1)

    image_size = image_size or inputs.shape[-2:]
    outputs = inputs.new_zeros(*inputs.shape[:-2], *image_size)

    def flatten_multi_index(multi_indices, shape):
        indices, *multi_indices = multi_indices
        if not multi_indices: return indices
        return (
            indices * functools.reduce(operator.mul, shape[1:]) +
            flatten_multi_index(multi_indices, shape[1:])
        )

    if interp_mode == "nearest":

        spatial_coords = torch.round(spatial_coords).long()

        lower_bound_masks = 0.0 <= spatial_coords
        upper_bound_masks = spatial_coords < inputs.new_tensor(image_size).flip(-1)

        range_masks = torch.all(lower_bound_masks & upper_bound_masks, dim=-1)
        masks = range_masks if valid_masks is None else valid_masks & range_masks

        masked_inputs = inputs[masks]
        masked_batch_coords = batch_coords[masks]
        masked_channel_coords = channel_coords[masks]
        masked_spatial_coords = spatial_coords[masks, ...]

        masked_spatial_coords_x, masked_spatial_coords_y = torch.unbind(masked_spatial_coords, dim=-1)

        multi_masked_indices = (
            masked_batch_coords,
            masked_channel_coords,
            masked_spatial_coords_y,
            masked_spatial_coords_x,
        )
        masked_indices = flatten_multi_index(multi_masked_indices, outputs.shape)
        masked_weights = torch.ones_like(masked_inputs)

    if interp_mode == "bilinear":

        min_spatial_coords = torch.floor(spatial_coords).long()
        max_spatial_coords = min_spatial_coords + 1

        lower_bound_masks = 0.0 <= min_spatial_coords
        upper_bound_masks = max_spatial_coords < inputs.new_tensor(image_size).flip(-1)

        range_masks = torch.all(lower_bound_masks & upper_bound_masks, dim=-1)
        masks = range_masks if valid_masks is None else valid_masks & range_masks

        masked_inputs = inputs[masks]
        masked_batch_coords = batch_coords[masks]
        masked_channel_coords = channel_coords[masks]
        masked_spatial_coords = spatial_coords[masks, ...]
        masked_min_spatial_coords = min_spatial_coords[masks, ...]
        masked_max_spatial_coords = max_spatial_coords[masks, ...]

        masked_spatial_coords_x, masked_spatial_coords_y = torch.unbind(masked_spatial_coords, dim=-1)
        masked_min_spatial_coords_x, masked_min_spatial_coords_y = torch.unbind(masked_min_spatial_coords, dim=-1)
        masked_max_spatial_coords_x, masked_max_spatial_coords_y = torch.unbind(masked_max_spatial_coords, dim=-1)

        masked_minmax_spatial_coords_x = torch.stack([masked_min_spatial_coords_x, masked_max_spatial_coords_x], dim=0)
        masked_minmax_spatial_coords_y = torch.stack([masked_min_spatial_coords_y, masked_max_spatial_coords_y], dim=0)

        masked_weights_x = 1.0 - torch.abs(masked_minmax_spatial_coords_x - masked_spatial_coords_x)
        masked_weights_y = 1.0 - torch.abs(masked_minmax_spatial_coords_y - masked_spatial_coords_y)

        multi_masked_indices = (
            masked_batch_coords.unsqueeze(0).unsqueeze(1),
            masked_channel_coords.unsqueeze(0).unsqueeze(1),
            masked_minmax_spatial_coords_y.unsqueeze(0),
            masked_minmax_spatial_coords_x.unsqueeze(1),
        )

        masked_indices = flatten_multi_index(multi_masked_indices, outputs.shape).flatten()
        masked_weights = (masked_weights_y.unsqueeze(0) * masked_weights_x.unsqueeze(1)).flatten()

        masked_inputs = masked_inputs.expand(2, 2, -1).flatten()

    outputs = outputs.flatten()
    weights = torch.zeros_like(outputs)

    # NOTE: `scatter_min` is more preferable?
    outputs.scatter_add_(index=masked_indices, src=masked_inputs * masked_weights, dim=0)
    weights.scatter_add_(index=masked_indices, src=masked_weights, dim=0)

    outputs = outputs / torch.clamp(weights, min=1e-6)
    outputs = outputs.reshape(*inputs.shape[:-2], *image_size)

    return outputs


def correlation(target_features, source_features, normalize=False, keepdim=False):
    if normalize:
        target_features = nn.functional.normalize(target_features, dim=1)
        source_features = nn.functional.normalize(source_features, dim=1)
    cost_volumes = torch.mean(target_features * source_features, dim=1, keepdim=keepdim)
    return cost_volumes


def groupwise_correlation(target_features, source_features, num_groups, normalize=False, keepdim=False):
    assert not target_features.shape[1] % num_groups and not source_features.shape[1] % num_groups
    target_features = target_features.reshape(*target_features.shape[:1], num_groups, -1, *target_features.shape[2:])
    source_features = source_features.reshape(*source_features.shape[:1], num_groups, -1, *source_features.shape[2:])
    if normalize:
        target_features = nn.functional.normalize(target_features, dim=2)
        source_features = nn.functional.normalize(source_features, dim=2)
    cost_volumes = torch.mean(target_features * source_features, dim=2, keepdim=keepdim)
    return cost_volumes


def clip_lines_to_front(lines, epsilon=1e-6):

    points_1, points_2 = torch.unbind(lines, dim=-2)
    depths_1, depths_2 = points_1[..., -1:], points_2[..., -1:]

    points_1, points_2 = (
        torch.where(depths_1 > depths_2, points_1, points_2),
        torch.where(depths_1 > depths_2, points_2, points_1),
    )
    depths_1, depths_2 = (
        torch.where(depths_1 > depths_2, depths_1, depths_2),
        torch.where(depths_1 > depths_2, depths_2, depths_1),
    )

    weights = depths_1 / torch.clamp(depths_1 - depths_2, min=epsilon)
    weights = torch.clamp(weights, max=1.0)

    points_2 = points_1 + (points_2 - points_1) * weights
    lines = torch.stack([points_1, points_2], dim=-2)

    masks = points_1[..., -1] > 0

    return lines, masks


def project_box_3d(box_3d, line_indices, intrinsic_matrix, epsilon=1e-6):

    lines = box_3d[..., line_indices, :]
    lines, masks = clip_lines_to_front(lines, epsilon)

    lines = lines @ intrinsic_matrix.T
    lines = lines[..., :-1] / torch.clamp(lines[..., -1:], min=epsilon)

    if torch.any(masks):

        points = lines[masks, ...].flatten(-3, -2)
        box_2d = torch.stack([
            torch.min(points, dim=-2).values,
            torch.max(points, dim=-2).values,
        ], dim=-2)

    else:

        # NOTE: how a box behind the camera should be projected?
        box_2d = box_3d.new_zeros(*box_3d.shape[:-2], 2, 2)

    return box_2d
