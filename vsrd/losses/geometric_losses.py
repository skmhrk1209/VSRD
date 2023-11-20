# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import torch
import torch.nn as nn

from . import utils


@utils.reduced
def rotation_consistency_loss(source_extrinsic_matrices, target_extrinsic_matrices, epsilon=1e-6):

    cycle_extrinsic_matrices = target_extrinsic_matrices @ source_extrinsic_matrices
    identity_matrices = torch.eye(3).to(cycle_extrinsic_matrices)

    def consistency_loss(rotation_matrices):
        consistency_losses = nn.functional.mse_loss(
            input=rotation_matrices,
            target=identity_matrices,
            reduction="none",
        )
        consistency_losses = torch.mean(consistency_losses, dim=(-2, -1))
        return consistency_losses

    rotation_consistency_losses = (
        consistency_loss(cycle_extrinsic_matrices[..., :3, :3]) / (
            consistency_loss(source_extrinsic_matrices[..., :3, :3]) +
            consistency_loss(target_extrinsic_matrices[..., :3, :3]) +
            epsilon
        )
    )

    return rotation_consistency_losses


@utils.reduced
def translation_consistency_loss(source_extrinsic_matrices, target_extrinsic_matrices, epsilon=1e-6):

    cycle_extrinsic_matrices = target_extrinsic_matrices @ source_extrinsic_matrices
    zero_vectors = torch.zeros(3).to(cycle_extrinsic_matrices)

    def consistency_loss(translation_vectors):
        consistency_losses = nn.functional.mse_loss(
            input=translation_vectors,
            target=zero_vectors,
            reduction="none",
        )
        consistency_losses = torch.mean(consistency_losses, dim=-1)
        return consistency_losses

    translation_consistency_losses = (
        consistency_loss(cycle_extrinsic_matrices[..., :3, 3]) / (
            consistency_loss(source_extrinsic_matrices[..., :3, 3]) +
            consistency_loss(target_extrinsic_matrices[..., :3, 3]) +
            epsilon
        )
    )

    return translation_consistency_losses


@utils.reduced
def sampson_epipolar_distance(keypoints_1, keypoints_2, fundamental_matrices):

    keypoints_1 = nn.functional.pad(keypoints_1, (0, 1), mode="constant", value=1.0)
    keypoints_2 = nn.functional.pad(keypoints_2, (0, 1), mode="constant", value=1.0)

    epipolar_lines_2 = keypoints_1 @ fundamental_matrices.transpose(-2, -1)
    epipolar_lines_1 = keypoints_2 @ fundamental_matrices

    algebraic_errors = torch.sum(keypoints_2 * epipolar_lines_2, dim=-1) ** 2.0

    gradient_norms_2 = torch.sum(epipolar_lines_2[..., :2] ** 2.0, dim=-1)
    gradient_norms_1 = torch.sum(epipolar_lines_1[..., :2] ** 2.0, dim=-1)

    sampson_epipolar_distances = algebraic_errors / (gradient_norms_2 + gradient_norms_1)

    return sampson_epipolar_distances
