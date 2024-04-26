import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2 as cv
import skimage

from .. import utils


class Resizer(nn.Module):

    def __init__(
        self,
        image_size,
        image_interp_mode="bilinear",
        masks_interp_mode="nearest",
    ):
        super().__init__()
        self.image_size = image_size
        self.image_interp_mode = image_interp_mode
        self.masks_interp_mode = masks_interp_mode

    def forward(self, image, masks=None, intrinsic_matrix=None, **kwargs):

        assert masks is None or image.shape[-2:] == masks.shape[-2:]

        scale_factor = np.divide(self.image_size, image.shape[-2:])

        image = utils.unvectorize(nn.functional.interpolate)(
            input=image,
            size=self.image_size,
            mode=self.image_interp_mode,
        )

        if masks is not None:

            if len(masks):
                masks = utils.unvectorize(nn.functional.interpolate)(
                    input=masks,
                    size=self.image_size,
                    mode=self.masks_interp_mode,
                )
            else:
                masks = masks.new_empty(*masks.shape[:-2], *self.image_size)

        if intrinsic_matrix is not None:

            intrinsic_matrix = intrinsic_matrix.new_tensor([
                [scale_factor[-1], 0.0, 0.0],
                [0.0, scale_factor[-2], 0.0],
                [0.0, 0.0, 1.0],
            ]) @ intrinsic_matrix

        return dict(
            kwargs,
            image=image,
            masks=masks,
            intrinsic_matrix=intrinsic_matrix,
        )


class Cropper(nn.Module):

    def __init__(self, position=None, image_size=None):
        super().__init__()
        self.position = position
        self.image_size = image_size

    def forward(self, image, masks=None, intrinsic_matrix=None, crop_box=None, **kwargs):

        if crop_box is not None:
            position, _ = map(reversed, torch.unbind(crop_box.long(), dim=-2))
            image_size = reversed(torch.sub(*reversed(torch.unbind(crop_box.long(), dim=-2))))
        else:
            position = self.position
            image_size = self.image_size

        assert masks is None or image.shape[-2:] == masks.shape[-2:]

        image = torchvision.transforms.functional.crop(image, *position, *image_size)

        if masks is not None:

            masks = torchvision.transforms.functional.crop(masks, *self.position, *self.image_size)

        if intrinsic_matrix is not None:

            intrinsic_matrix = intrinsic_matrix.new_tensor([
                [1.0, 0.0, -self.position[-1]],
                [0.0, 1.0, -self.position[-2]],
                [0.0, 0.0, 1.0],
            ]) @ intrinsic_matrix

        return dict(
            kwargs,
            image=image,
            masks=masks,
            intrinsic_matrix=intrinsic_matrix,
        )


class RandomHorizontalFlipper(nn.Module):

    def __init__(self, probability=0.5):
        super().__init__()
        self.bernoulli = torch.distributions.Bernoulli(probability)
        self.update_params()

    def update_params(self):
        self.bernoulli_variable = self.bernoulli.sample()

    def forward(self, image, masks=None, intrinsic_matrix=None, **kwargs):

        if self.bernoulli_variable:

            image = image.flip(-1)

            if masks is not None:

                masks = masks.flip(-1)

            if intrinsic_matrix is not None:

                intrinsic_matrix = intrinsic_matrix.new_tensor([
                    [-1.0, 0.0, image.shape[-1] - 1],
                    [0.0, -1.0, image.shape[-2] - 1],
                    [0.0, 0.0, 1.0],
                ]) @ intrinsic_matrix

        return dict(
            kwargs,
            image=image,
            masks=masks,
            intrinsic_matrix=intrinsic_matrix,
        )


class BoxGenerator(nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, masks, **kwargs):

        @utils.torch_function
        def where(*args, **kwargs):
            return np.where(*args, **kwargs)

        if len(masks):
            binary_masks = masks > self.threshold
            boxes_2d = torch.stack([
                torch.stack([
                    torch.stack(list(map(torch.min, where(binary_mask)))),
                    torch.stack(list(map(torch.max, where(binary_mask)))),
                ], dim=0)
                for binary_mask in binary_masks
            ], dim=0).flip(-1).to(masks)
        else:
            boxes_2d = masks.new_empty(*masks.shape[:-2], 2, 2)

        return dict(
            kwargs,
            masks=masks,
            boxes_2d=boxes_2d,
        )

        binary_masks = masks > self.threshold
        boxes_2d = torchvision.ops.masks_to_boxes(binary_masks)
        boxes_2d = boxes_2d.unflatten(-1, (2, 2))

        return dict(
            kwargs,
            masks=masks,
            boxes_2d=boxes_2d,
        )


class MaskAreaFilter(nn.Module):

    def __init__(self, min_mask_area, threshold=0.5):
        super().__init__()
        self.min_mask_area = min_mask_area
        self.threshold = threshold

    def forward(self, masks, labels, boxes_3d, instance_ids, **kwargs):

        mask_areas = torch.sum(masks > self.threshold, dim=(-2, -1))
        instance_mask = mask_areas >= self.min_mask_area

        masks = masks[instance_mask, ...]
        labels = labels[instance_mask, ...]
        boxes_3d = boxes_3d[instance_mask, ...]
        instance_ids = instance_ids[instance_mask, ...]

        return dict(
            kwargs,
            masks=masks,
            labels=labels,
            boxes_3d=boxes_3d,
            instance_ids=instance_ids,
        )


class BoxSizeFilter(nn.Module):

    def __init__(self, min_box_size):
        super().__init__()
        self.min_box_size = min_box_size

    def forward(self, masks, labels, boxes_3d, boxes_2d, instance_ids, **kwargs):

        box_sizes = torch.min(-torch.sub(*torch.unbind(boxes_2d, dim=-2)), dim=-1).values
        instance_mask = box_sizes >= self.min_box_size

        masks = masks[instance_mask, ...]
        labels = labels[instance_mask, ...]
        boxes_3d = boxes_3d[instance_mask, ...]
        boxes_2d = boxes_2d[instance_mask, ...]
        instance_ids = instance_ids[instance_mask, ...]

        return dict(
            kwargs,
            masks=masks,
            labels=labels,
            boxes_3d=boxes_3d,
            boxes_2d=boxes_2d,
            instance_ids=instance_ids,
        )


class SoftRasterizer(nn.Module):

    def __init__(self, threshold=0.5, temperature=10.0):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

    @utils.torch_function
    def make_polygon(self, mask):
        mask = mask > self.threshold
        mask = skimage.img_as_ubyte(mask)
        polygons, _ = cv.findContours(
            image=mask,
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_SIMPLE,
        )
        polygon = max(polygons, key=cv.contourArea)
        polygon = polygon.squeeze(-2)
        return polygon

    @utils.torch_function
    def make_binary_mask(self, polygon, image_size):
        mask = np.zeros(image_size, dtype=np.uint8)
        mask = cv.fillPoly(
            img=mask,
            pts=[polygon],
            color=(1 << 8) - 1,
            lineType=cv.LINE_8,
        )
        mask = skimage.img_as_bool(mask)
        return mask

    def make_distance_map(self, polygons, image_size):
        # [HW, 2]
        positions = list(reversed(torch.meshgrid(*map(torch.arange, image_size), indexing="ij")))
        positions = torch.stack(list(map(torch.flatten, positions)), dim=-1)
        # [B, N, 2]
        prev_vertices, next_vertices = polygons, torch.roll(polygons, shifts=-1, dims=-2)
        # [B, 1, N, 2]
        polysides = next_vertices.unsqueeze(-3) - prev_vertices.unsqueeze(-3)
        # [B, HW, N, 2]
        positions = positions.unsqueeze(-2) - prev_vertices.unsqueeze(-3)
        # [B, HW, N, 1]
        ratios = (
            torch.sum(polysides * positions, dim=-1, keepdim=True) /
            (torch.sum(polysides * polysides, dim=-1, keepdim=True) + 1e-6)
        )
        # [B, HW, N, 2]
        normals = positions - polysides * torch.clamp(ratios, 0.0, 1.0)
        # [B, HW, N]
        distances = torch.linalg.norm(normals, dim=-1)
        # [B, HW]
        distances = torch.min(distances, dim=-1).values
        # [B, H, W]
        distance_maps = distances.unflatten(-1, image_size)
        return distance_maps

    def forward(self, masks, **kwargs):

        if len(masks):

            polygons = list(map(self.make_polygon, masks))

            binary_masks = torch.stack([
                self.make_binary_mask(polygon, masks.shape[-2:])
                for polygon in polygons
            ], dim=0)

            distance_maps = torch.stack([
                self.make_distance_map(polygon, masks.shape[-2:])
                for polygon in polygons
            ], dim=0)

            sdf_maps = torch.where(binary_masks, distance_maps, -distance_maps)
            soft_masks = torch.sigmoid(sdf_maps / self.temperature)

        else:
            soft_masks = torch.empty_like(masks)

        return dict(
            kwargs,
            masks=masks,
            hard_masks=masks,
            soft_masks=soft_masks,
        )


class MaskRefiner(nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @utils.torch_function
    def make_polygon(self, mask):
        mask = mask > self.threshold
        mask = skimage.img_as_ubyte(mask)
        polygons, _ = cv.findContours(
            image=mask,
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_SIMPLE,
        )
        polygon = max(polygons, key=cv.contourArea)
        polygon = polygon.squeeze(-2)
        return polygon

    @utils.torch_function
    def make_mask(self, polygon, image_size):
        mask = np.zeros(image_size, dtype=np.uint8)
        mask = cv.fillPoly(
            img=mask,
            pts=[polygon.astype(np.int64)],
            color=(1 << 8) - 1,
            lineType=cv.LINE_8,
        )
        mask = skimage.img_as_float32(mask)
        return mask

    @utils.vectorize
    def refine_mask(self, mask):
        polygon = self.make_polygon(mask)
        mask = self.make_mask(polygon, mask.shape[-2:])
        return mask

    def forward(self, masks, **kwargs):
        if masks.numel():
            masks = self.refine_mask(masks)
        return dict(kwargs, masks=masks)


class BoxJitter(nn.Module):

    def __init__(self, scale_range):
        super().__init__()
        self.scale_distribution = torch.distributions.uniform.Uniform(*scale_range)

    def forward(self, crop_box, **kwargs):
        sizes = torch.sub(*reversed(torch.unbind(crop_box, dim=-2)))
        scales = self.scale_distribution.sample(crop_box.shape)
        crop_box = crop_box + sizes * scales
        return dict(kwargs, crop_box=crop_box)
