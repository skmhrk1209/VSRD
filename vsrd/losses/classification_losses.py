# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import torch
import torch.nn as nn

from . import utils


@utils.reduced
def cross_entropy(inputs, targets, dim=None, keepdim=False, epsilon=1e-6):
    inputs = nn.functional.hardtanh(inputs, 0.0 + epsilon, 1.0 - epsilon)
    losses = -targets * torch.log(inputs)
    if dim: losses = torch.sum(losses, dim=dim, keepdim=keepdim)
    return losses


@utils.reduced
def binary_cross_entropy(inputs, targets, epsilon=1e-6):
    losses = (
        cross_entropy(
            inputs=inputs,
            targets=targets,
            epsilon=epsilon,
            reduction="none",
        ) +
        cross_entropy(
            inputs=1.0 - inputs,
            targets=1.0 - targets,
            epsilon=epsilon,
            reduction="none",
        )
    )
    return losses


@utils.reduced
def kl_divergence(inputs, targets, dim=None, keepdim=False, epsilon=1e-6):
    inputs = nn.functional.hardtanh(inputs, 0.0 + epsilon, 1.0 - epsilon)
    targets = nn.functional.hardtanh(targets, 0.0 + epsilon, 1.0 - epsilon)
    losses = -targets * (torch.log(inputs) - torch.log(targets))
    if dim: losses = torch.sum(losses, dim=dim, keepdim=keepdim)
    return losses


@utils.reduced
def binary_kl_divergence(inputs, targets, epsilon=1e-6):
    losses = (
        kl_divergence(
            inputs=inputs,
            targets=targets,
            epsilon=epsilon,
            reduction="none",
        ) +
        kl_divergence(
            inputs=1.0 - inputs,
            targets=1.0 - targets,
            epsilon=epsilon,
            reduction="none",
        )
    )
    return losses


@utils.reduced
def js_divergence(inputs, targets, dim=None, keepdim=False, epsilon=1e-6):
    means = inputs * 0.5 + targets * 0.5
    losses = (
        kl_divergence(
            inputs=means,
            targets=inputs,
            dim=dim,
            keepdim=keepdim,
            epsilon=epsilon,
            reduction="none",
        ) * 0.5 +
        kl_divergence(
            inputs=means,
            targets=targets,
            dim=dim,
            keepdim=keepdim,
            epsilon=epsilon,
            reduction="none",
        ) * 0.5
    )
    return losses


@utils.reduced
def binary_js_divergence(inputs, targets, epsilon=1e-6):
    losses = (
        js_divergence(
            inputs=inputs,
            targets=targets,
            epsilon=epsilon,
            reduction="none",
        ) +
        js_divergence(
            inputs=1.0 - inputs,
            targets=1.0 - targets,
            epsilon=epsilon,
            reduction="none",
        )
    )
    return losses


@utils.reduced
def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """ Focal Loss

    References:
        - [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
    """
    losses = (
        (1.0 - torch.abs(targets - alpha)) *                    # α_t
        torch.abs(targets - inputs) ** gamma *                  # (1 - p_t)^γ
        binary_cross_entropy(inputs, targets, reduction="none") # -log(p_t)
    )
    return losses


@utils.reduced
def quality_focal_loss(inputs, targets, beta=2.0):
    """ Quality Focal Loss

    References:
        - [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388)
    """
    losses = (
        torch.abs(targets - inputs) ** beta *                   # |y - σ|^β
        binary_cross_entropy(inputs, targets, reduction="none") # -((1 - y)log(1 - σ) + ylog(σ))
    )
    return losses


@utils.reduced
def tversky_loss(inputs, targets, alpha=0.7, beta=0.3, epsilon=1.0):
    TP = torch.sum(inputs * targets, dim=(-2, -1))
    FN = torch.sum((1.0 - inputs) * targets, dim=(-2, -1))
    FP = torch.sum(inputs * (1.0 - targets), dim=(-2, -1))
    tversky_indices = (TP + epsilon) / (TP + alpha * FN + beta * FP + epsilon)
    tversky_losses = 1.0 - tversky_indices
    return tversky_losses


@utils.reduced
def focal_tversky_loss(inputs, targets, gamma=0.75, **kwargs):
    tversky_losses = tversky_loss(inputs, targets, **kwargs, reduction="none")
    focal_tversky_losses = tversky_losses ** gamma
    return focal_tversky_losses
