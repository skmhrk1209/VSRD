import torch
import numpy as np
import skimage
import cv2 as cv

from .. import utils
from .. import operations


@utils.torch_function
def draw_boxes_3d(image, boxes_3d, line_indices, intrinsic_matrix, *args, **kwargs):

    is_float = image.dtype.kind == "f"

    if is_float:
        image = skimage.img_as_ubyte(image)

    image = image.transpose(1, 2, 0)
    image = np.ascontiguousarray(image)

    # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552

    for box_3d in boxes_3d:

        lines = box_3d[line_indices, ...]
        lines, masks = utils.numpy_function(operations.clip_lines_to_front)(lines)

        lines = lines @ intrinsic_matrix.T
        lines = lines[..., :-1] / np.clip(lines[..., -1:], 1e-3, None)

        for (point_1, point_2) in lines[masks, ...]:

            image = cv.line(
                img=image,
                pt1=tuple(map(int, point_1)),
                pt2=tuple(map(int, point_2)),
                *args,
                **kwargs,
            )

    image = image.transpose(2, 0, 1)

    if is_float:
        image = skimage.img_as_float32(image)

    return image


@utils.torch_function
def draw_boxes_bev(image, boxes_3d, extents=((-50.0, 100.0), (50.0, 0.0)), *args, **kwargs):

    is_float = image.dtype.kind == "f"

    if is_float:
        image = skimage.img_as_ubyte(image)

    image = image.transpose(1, 2, 0)
    image = np.ascontiguousarray(image)

    # NOTE: use the KITTI-360 "evaluation" format instaed of the KITTI-360 "annotation" format
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552

    boxes_2d = np.mean(boxes_3d.reshape(-1, 2, 4, 3), axis=1)
    boxes_2d = (boxes_2d[..., [0, 2]] - extents[0]) / -np.subtract(*extents) * image.shape[:2][::-1]

    for box_2d in boxes_2d:

        for (point_1, point_2) in zip(box_2d, np.roll(box_2d, shift=-1, axis=0)):

            image = cv.line(
                img=image,
                pt1=tuple(map(int, point_1)),
                pt2=tuple(map(int, point_2)),
                *args,
                **kwargs,
            )

    for y in range(0, image.shape[0], image.shape[0] // 10):

        image = cv.line(
            img=image,
            pt1=(0, y),
            pt2=(image.shape[1], y),
            color=(128, 128, 128),
        )

    for x in range(0, image.shape[1], image.shape[1] // 10):

        image = cv.line(
            img=image,
            pt1=(x, 0),
            pt2=(x, image.shape[0]),
            color=(128, 128, 128),
        )

    image = image.transpose(2, 0, 1)

    if is_float:
        image = skimage.img_as_float32(image)

    return image


@utils.torch_function
def draw_boxes_2d(image, boxes_2d, *args, **kwargs):

    is_float = image.dtype.kind == "f"

    if is_float:
        image = skimage.img_as_ubyte(image)

    image = image.transpose(1, 2, 0)
    image = np.ascontiguousarray(image)

    for point_1, point_2 in boxes_2d:

        image = cv.rectangle(
            img=image,
            pt1=tuple(map(int, point_1)),
            pt2=tuple(map(int, point_2)),
            *args,
            **kwargs,
        )

    image = image.transpose(2, 0, 1)

    if is_float:
        image = skimage.img_as_float32(image)

    return image


@utils.torch_function
def draw_points_2d(image, points_2d, *args, **kwargs):

    is_float = image.dtype.kind == "f"

    if is_float:
        image = skimage.img_as_ubyte(image)

    image = image.transpose(1, 2, 0)
    image = np.ascontiguousarray(image)

    for point_2d in points_2d:

        image = cv.circle(
            img=image,
            center=tuple(map(int, point_2d)),
            *args,
            **kwargs,
        )

    image = image.transpose(2, 0, 1)

    if is_float:
        image = skimage.img_as_float32(image)

    return image


def draw_masks(image, masks, colors=None, weight=0.5):

    is_uint8 = image.dtype == torch.uint8

    if is_uint8:
        image = image.float() / ((1 << 8) - 1)

    if colors is None:
        colors = torch.rand(len(masks), 3).to(masks)
        colors = colors / torch.max(colors, dim=-1, keepdim=True).values

    colors = colors.T.reshape(3, -1, 1, 1)
    masks = torch.sum(masks * colors, dim=1)

    image = image + masks * weight
    image = torch.clamp(image, 0.0, 1.0)

    if is_uint8:
        image = (image * ((1 << 8) - 1)).byte()

    return image
