# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import torch
import torch.nn as nn


def rotation_matrix_y(cos, sin):
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)
    rotation_matrices = torch.stack([
        torch.stack([ cos, zero,  sin], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([-sin, zero,  cos], dim=-1),
    ], dim=-2)
    return rotation_matrices


class BoxParameters3D(nn.Module):

    def __init__(
        self,
        batch_size,
        num_instances,
        num_features=256,
        location_range=[
            [-50.0, 1.55 - 1.75 / 2.0 - 5.0, 000.0],
            [+50.0, 1.55 - 1.75 / 2.0 + 5.0, 100.0],
        ],
        dimension_range=[
            [0.75, 0.75, 1.5],
            [1.00, 1.00, 2.5],
        ],
    ):
        super().__init__()

        self.register_parameter(
            "locations",
            nn.Parameter(torch.zeros(batch_size, num_instances, 3)),
        )
        self.register_parameter(
            "dimensions",
            nn.Parameter(torch.zeros(batch_size, num_instances, 3)),
        )
        self.register_parameter(
            "orientations",
            nn.Parameter(torch.tensor([1.0, 0.0]).repeat(batch_size, num_instances, 1)),
        )
        self.register_parameter(
            "embeddings",
            nn.Parameter(torch.rand(num_features).repeat(batch_size, num_instances, 1)),
        )

        self.register_buffer(
            "location_range",
            torch.as_tensor(location_range),
        )
        self.register_buffer(
            "dimension_range",
            torch.as_tensor(dimension_range),
        )

    def decode_location(self, locations):
        locations = torch.lerp(*self.location_range, torch.sigmoid(locations))
        return locations

    def decode_dimension(self, dimensions):
        dimensions = torch.lerp(*self.dimension_range, torch.sigmoid(dimensions))
        return dimensions

    def decode_orientation(self, orientations):
        orientations = nn.functional.normalize(orientations, dim=-1)
        rotation_matrices = rotation_matrix_y(*torch.unbind(orientations, dim=-1))
        return rotation_matrices

    @staticmethod
    def decode_box_3d(locations, dimensions, orientations):
        # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
        # NOTE: the KITTI-360 "annotation" format is different from the KITTI-360 "evaluation" format
        # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
        # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552
        boxes = dimensions.new_tensor([
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, +1.0, +1.0],
            [+1.0, +1.0, +1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],
        ]) * dimensions.unsqueeze(-2)
        boxes = boxes @ orientations.transpose(-2, -1)
        boxes = boxes + locations.unsqueeze(-2)
        return boxes

    @staticmethod
    def encode_box_3d(boxes_3d):

        locations = torch.mean(boxes_3d, dim=-2)

        widths = torch.mean(torch.norm(torch.sub(
            boxes_3d[..., [1, 2, 6, 5], :],
            boxes_3d[..., [0, 3, 7, 4], :],
        ), dim=-1), dim=-1)

        heights = torch.mean(torch.norm(torch.sub(
            boxes_3d[..., [4, 5, 6, 7], :],
            boxes_3d[..., [0, 1, 2, 3], :],
        ), dim=-1), dim=-1)

        lengths = torch.mean(torch.norm(torch.sub(
            boxes_3d[..., [1, 0, 4, 5], :],
            boxes_3d[..., [2, 3, 7, 6], :],
        ), dim=-1), dim=-1)

        dimensions = torch.stack([widths, heights, lengths], dim=-1) / 2.0

        orientations = torch.mean(torch.sub(
            boxes_3d[..., [1, 0, 4, 5], :],
            boxes_3d[..., [2, 3, 7, 6], :],
        ), dim=-2)

        orientations = nn.functional.normalize(orientations[..., [2, 0]], dim=-1)
        orientations = rotation_matrix_y(*torch.unbind(orientations, dim=-1))

        return locations, dimensions, orientations

    def forward(self):

        # decode box parameters
        locations = self.decode_location(self.locations)
        dimensions = self.decode_dimension(self.dimensions)
        orientations = self.decode_orientation(self.orientations)

        # decode 3D bounding box
        boxes_3d = self.decode_box_3d(
            locations=locations,
            dimensions=dimensions,
            orientations=orientations,
        )

        outputs = dict(
            boxes_3d=boxes_3d,
            locations=locations,
            dimensions=dimensions,
            orientations=orientations,
            embeddings=self.embeddings,
        )

        return outputs
