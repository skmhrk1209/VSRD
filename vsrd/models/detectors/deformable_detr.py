import operator
import itertools

import torch
import torch.nn as nn
import torchvision
import scipy as sp

from transformers import DeformableDetrForObjectDetection
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrLoss,
    DeformableDetrHungarianMatcher,
    DeformableDetrMLPPredictionHead,
    generalized_box_iou,
    inverse_sigmoid,
)
from transformers.image_transforms import (
    center_to_corners_format,
)

from ... import utils


def rotation_matrix_y(cos, sin):
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)
    rotation_matrices = torch.stack([
        torch.stack([ cos, zero,  sin], dim=-1),
        torch.stack([zero,  one, zero], dim=-1),
        torch.stack([-sin, zero,  cos], dim=-1),
    ], dim=-2)
    return rotation_matrices


class DistributedDeformableDetrLoss(DeformableDetrLoss):

    def forward(self, outputs, targets):

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(target["class_labels"]) for target in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # (Niels): comment out function below, distributed training to be added
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes /= torch.distributed.get_world_size()

        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = dict(sum([list(self.get_loss(loss, outputs, targets, indices, num_boxes).items()) for loss in self.losses], []))

        return losses


class DeformableDetr3DHungarianMatcher(DeformableDetrHungarianMatcher):

    def __init__(self, *args, location_cost=1.0, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.location_cost = location_cost
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets):

        batch_size, num_queries = outputs["logits"].shape[:2]
        num_targets = [len(target["class_labels"]) for target in targets]

        # We flatten to compute the cost matrices in a batch
        pred_scores = torch.sigmoid(outputs["logits"].flatten(0, 1))  # [batch_size * num_queries, num_classes]
        pred_boxes_2d = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        pred_locations = outputs["locations"].flatten(0, 1)

        # Also concat the target labels and boxes
        target_labels = torch.cat([target["class_labels"] for target in targets])
        target_boxes_2d = torch.cat([target["boxes"] for target in targets])
        target_locations = torch.cat([target["locations"] for target in targets])

        # Compute the classification cost.
        negative_class_costs = (1.0 - self.alpha) * ((pred_scores - 0.0) ** self.gamma) * -torch.log((1.0 - pred_scores) + 1e-8)
        positive_class_costs = (self.alpha - 0.0) * ((1.0 - pred_scores) ** self.gamma) * -torch.log((pred_scores - 0.0) + 1e-8)
        class_cost_matrix = positive_class_costs[..., target_labels] - negative_class_costs[..., target_labels]

        # Compute the L1 cost between boxes
        bbox_cost_matrix = torch.cdist(pred_boxes_2d, target_boxes_2d, p=1)

        # Compute the giou cost between boxes
        giou_cost_matrix = -generalized_box_iou(
            boxes1=center_to_corners_format(pred_boxes_2d),
            boxes2=center_to_corners_format(target_boxes_2d),
        )

        # Compute the L1 cost between locations
        location_cost_matrix = torch.cdist(pred_locations, target_locations, p=1)

        # Final cost matrix
        cost_matrix = (
            class_cost_matrix * self.class_cost +
            bbox_cost_matrix * self.bbox_cost +
            giou_cost_matrix * self.giou_cost +
            location_cost_matrix * self.location_cost
        )
        cost_matrices = cost_matrix.reshape(batch_size, num_queries, -1)

        matched_indices = [
            utils.torch_function(sp.optimize.linear_sum_assignment)(cost_matrices[batch_index])
            for batch_index, cost_matrices in enumerate(torch.split(cost_matrices, num_targets, dim=-1))
        ]

        return matched_indices


class DeformableDETR3D(DeformableDetrForObjectDetection):

    def __init__(
        self,
        config,
        depth_range=[000.0, 100.0],
        dimension_range=[
            [0.75, 0.75, 1.5],
            [1.00, 1.00, 2.5],
        ],
        image_normalizer=torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ):
        super().__init__(config)

        assert not config.two_stage
        assert not config.with_box_refine

        del self.class_embed, self.bbox_embed

        self.image_normalizer = image_normalizer

        self.classification_head = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            # NOTE: DETR: softmax, Deformable DETR: sigmoid
            output_dim=config.num_labels,
            num_layers=3,
        )
        self.box_2d_regression_head = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=4,
            num_layers=3,
        )
        self.box_3d_regression_head = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=8,
            num_layers=3,
        )
        self.confidence_prediction_head = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=1,
            num_layers=3,
        )

        self.criterion = DistributedDeformableDetrLoss(
            matcher=DeformableDetr3DHungarianMatcher(
                class_cost=config.class_cost,
                bbox_cost=config.bbox_cost,
                giou_cost=config.giou_cost,
                location_cost=config.location_cost,
            ),
            num_classes=config.num_labels,
            focal_alpha=config.focal_alpha,
            losses=["labels", "boxes", "cardinality"],
        )

        self.register_buffer(
            "depth_range",
            torch.as_tensor(depth_range),
        )
        self.register_buffer(
            "dimension_range",
            torch.as_tensor(dimension_range),
        )

        self.loss_weights = dict(
            loss_ce=config.class_loss_coefficient,
            loss_bbox=config.bbox_loss_coefficient,
            loss_giou=config.giou_loss_coefficient,
            cardinality_error=0.0,
        )

        self.post_init()

        assert len(list(itertools.chain(
            self.backbone_parameters(),
            self.transformer_parameters(),
            self.classification_head_parameters(),
            self.box_2d_regression_head_parameters(),
            self.box_3d_regression_head_parameters(),
            self.confidence_prediction_head_parameters(),
            self.reference_points_parameters(),
            self.sampling_offsets_parameters(),
        ))) == len(list(self.parameters()))

    def backbone_named_parameters(self):
        yield from (
            (f"model.backbone.{name}", parameter)
            for name, parameter in self.model.backbone.named_parameters()
        )

    def backbone_parameters(self):
        yield from map(operator.itemgetter(1), self.backbone_named_parameters())

    def classification_head_named_parameters(self):
        yield from (
            (f"classification_head.{name}", parameter)
            for name, parameter in self.classification_head.named_parameters()
        )

    def classification_head_parameters(self):
        yield from map(operator.itemgetter(1), self.classification_head_named_parameters())

    def box_2d_regression_head_named_parameters(self):
        yield from (
            (f"box_2d_regression_head.{name}", parameter)
            for name, parameter in self.box_2d_regression_head.named_parameters()
        )

    def box_2d_regression_head_parameters(self):
        yield from map(operator.itemgetter(1), self.box_2d_regression_head_named_parameters())

    def box_3d_regression_head_named_parameters(self):
        yield from (
            (f"box_3d_regression_head.{name}", parameter)
            for name, parameter in self.box_3d_regression_head.named_parameters()
        )

    def box_3d_regression_head_parameters(self):
        yield from map(operator.itemgetter(1), self.box_3d_regression_head_named_parameters())

    def confidence_prediction_head_named_parameters(self):
        yield from (
            (f"confidence_prediction_head.{name}", parameter)
            for name, parameter in self.confidence_prediction_head.named_parameters()
        )

    def confidence_prediction_head_parameters(self):
        yield from map(operator.itemgetter(1), self.confidence_prediction_head_named_parameters())

    def transformer_named_parameters(self, include_reference_points=False, include_sampling_offsets=False):
        backbone_names = list(dict(self.backbone_named_parameters()))
        classification_head_names = list(dict(self.classification_head_named_parameters()))
        box_2d_regression_head_names = list(dict(self.box_2d_regression_head_named_parameters()))
        box_3d_regression_head_names = list(dict(self.box_3d_regression_head_named_parameters()))
        confidence_prediction_head_names = list(dict(self.confidence_prediction_head_named_parameters()))
        reference_points_names = (include_reference_points and []) or list(dict(self.reference_points_named_parameters()))
        sampling_offsets_names = (include_sampling_offsets and []) or list(dict(self.sampling_offsets_named_parameters()))
        for name, parameter in self.named_parameters():
            if all(name not in names for names in [
                backbone_names,
                classification_head_names,
                box_2d_regression_head_names,
                box_3d_regression_head_names,
                confidence_prediction_head_names,
                reference_points_names,
                sampling_offsets_names,
            ]):
                yield name, parameter

    def transformer_parameters(self):
        yield from map(operator.itemgetter(1), self.transformer_named_parameters())

    def reference_points_named_parameters(self):
        yield from (
            (f"model.reference_points.{name}", parameter)
            for name, parameter in self.model.reference_points.named_parameters()
        )

    def reference_points_parameters(self):
        yield from map(operator.itemgetter(1), self.reference_points_named_parameters())

    def sampling_offsets_named_parameters(self):
        yield from itertools.chain(
            itertools.chain.from_iterable(
                (
                    (f"model.encoder.layers.{layer_index}.self_attn.sampling_offsets.{name}", parameter)
                    for name, parameter in encoder_layer.self_attn.sampling_offsets.named_parameters()
                )
                for layer_index, encoder_layer in enumerate(self.model.encoder.layers)
            ),
            itertools.chain.from_iterable(
                (
                    (f"model.decoder.layers.{layer_index}.encoder_attn.sampling_offsets.{name}", parameter)
                    for name, parameter in decoder_layer.encoder_attn.sampling_offsets.named_parameters()
                )
                for layer_index, decoder_layer in enumerate(self.model.decoder.layers)
            ),
        )

    def sampling_offsets_parameters(self):
        yield from map(operator.itemgetter(1), self.sampling_offsets_named_parameters())

    def decode_location(
        self,
        locations,
        depths,
        image_size,
        intrinsic_matrices,
        extrinsic_matrices,
    ):
        locations = torch.sigmoid(locations) * image_size.flip(-1)
        locations = nn.functional.pad(locations, (0, 1), mode="constant", value=1.0)
        locations = locations @ torch.linalg.inv(intrinsic_matrices).transpose(-2, -1)
        locations = locations * torch.lerp(*self.depth_range, torch.sigmoid(depths))
        locations = nn.functional.pad(locations, (0, 1), mode="constant", value=1.0)
        locations = locations @ torch.linalg.inv(extrinsic_matrices).transpose(-2, -1)
        locations = locations[..., :-1] / locations[..., -1:]
        return locations

    def decode_dimension(self, dimensions):
        dimensions = torch.lerp(*self.dimension_range, torch.sigmoid(dimensions))
        return dimensions

    def decode_orientation(self, orientations, locations):
        orientations = nn.functional.normalize(orientations, dim=-1)
        locations = nn.functional.normalize(locations[..., [2, 0]], dim=-1)
        rotation_matrices = (
            rotation_matrix_y(*torch.unbind(orientations, dim=-1)) @
            rotation_matrix_y(*torch.unbind(locations, dim=-1))
        )
        return rotation_matrices

    @staticmethod
    def decode_box_3d(locations, dimensions, orientations):
        # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
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

    def forward(self, images, intrinsic_matrices, extrinsic_matrices):

        images = self.image_normalizer(images)

        outputs = self.model(images)

        logits = self.classification_head(outputs.last_hidden_state)
        boxes_2d = self.box_2d_regression_head(outputs.last_hidden_state)
        boxes_3d = self.box_3d_regression_head(outputs.last_hidden_state)
        confidences = torch.sigmoid(self.confidence_prediction_head(outputs.last_hidden_state))

        # image size for box unnormalization
        image_size = images.new_tensor(images.shape[-2:])

        boxes_2d = torch.sigmoid(boxes_2d)
        boxes_2d = center_to_corners_format(boxes_2d)
        boxes_2d = boxes_2d.unflatten(-1, (2, 2)) * image_size.flip(-1)

        # 7-DoF 3D bounding box
        locations, depths, dimensions, orientations = torch.split(boxes_3d, (2, 1, 3, 2), dim=-1)

        # box head predicts bounding boxes as relative offsets w.r.t. the reference points
        reference_points = outputs.intermediate_reference_points[:, -2, ...]
        reference_points = inverse_sigmoid(reference_points)
        locations = reference_points + locations

        # decode box parameters
        locations = self.decode_location(locations, depths, image_size, intrinsic_matrices, extrinsic_matrices)
        dimensions = self.decode_dimension(dimensions)
        orientations = self.decode_orientation(orientations, locations)

        # decode 3D bounding box
        boxes_3d = self.decode_box_3d(
            locations=locations,
            dimensions=dimensions,
            orientations=orientations,
        )

        outputs = dict(
            logits=logits,
            boxes_2d=boxes_2d,
            boxes_3d=boxes_3d,
            locations=locations,
            dimensions=dimensions,
            orientations=orientations,
            confidences=confidences,
        )

        return outputs
