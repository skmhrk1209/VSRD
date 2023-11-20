# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import torch
import torch.nn as nn


def norm(inputs, *args, epsilon=1e-6, **kwargs):
    return torch.sqrt(torch.sum(inputs ** 2.0, *args, **kwargs) + epsilon)


def box(dimension):

    def sdf(positions):
        positions = abs(positions) - dimension
        distances = (
            norm(nn.functional.relu(positions), dim=-1, keepdim=True) -
            nn.functional.relu(-torch.max(positions, dim=-1, keepdim=True).values)
        )
        return distances

    return sdf


def translation(sdf, translation_vector):

    def wrapper(positions):
        distances = sdf(positions - translation_vector)
        return distances

    return wrapper


def rotation(sdf, rotation_matrix):

    def wrapper(positions):
        distances = sdf(positions @ rotation_matrix)
        return distances

    return wrapper


def hard_union(sdfs):

    def wrapper(positions):
        distances = torch.stack([sdf(positions) for sdf in sdfs], dim=0)
        distances = torch.min(distances, dim=0).values
        return distances

    return wrapper


def soft_union(sdfs):

    def wrapper(positions):
        distances = torch.stack([sdf(positions) for sdf in sdfs], dim=0)
        weights = nn.functional.softmin(distances, dim=0)
        distances = torch.sum(distances * weights, dim=0)
        return distances

    return wrapper
