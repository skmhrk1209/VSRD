import os
import json
import glob
import argparse
import functools
import multiprocessing

import tqdm
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pycocotools.mask


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

    dimensions = torch.stack([widths, heights, lengths], dim=-1)

    orientations = torch.mean(torch.sub(
        boxes_3d[..., [1, 0, 4, 5], :],
        boxes_3d[..., [2, 3, 7, 6], :],
    ), dim=-2)

    orientations = nn.functional.normalize(orientations[..., [2, 0]], dim=-1)
    orientations = torch.atan2(*reversed(torch.unbind(orientations, dim=-1)))

    return locations, dimensions, orientations


def save_prediction(filename, class_names, boxes_3d, boxes_2d, scores):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as file:

        for class_name, box_3d, box_2d, score in zip(class_names, boxes_3d, boxes_2d, scores):

            location, dimension, orientation = encode_box_3d(box_3d)

            # ================================================================
            # KITTI-3D definition
            location[..., 1] += dimension[..., 1] / 2.0
            dimension = dimension[..., [1, 0, 2]]
            ray_orientation = torch.atan2(*reversed(torch.unbind(location[..., [2, 0]], dim=-1)))
            global_orientation = orientation - np.pi / 2.0
            local_orientation = global_orientation - ray_orientation
            # ================================================================

            file.write(
                f"{class_name.capitalize()} "
                f"{0.0} "
                f"{0} "
                f"{local_orientation} "
                f"{' '.join(map(str, box_2d.flatten().tolist()))} "
                f"{' '.join(map(str, dimension.tolist()))} "
                f"{' '.join(map(str, location.tolist()))} "
                f"{global_orientation} "
                f"{score}\n"
            )


def convert_predictions(sequence, root_dirname, ckpt_dirname, class_names):

    prediction_dirname = os.path.join("predictions", os.path.basename(ckpt_dirname))
    prediction_filenames = sorted(glob.glob(os.path.join(root_dirname, prediction_dirname, sequence, "image_00", "data_rect", "*.json")))

    for prediction_filename in prediction_filenames:

        with open(prediction_filename) as file:
            prediction = json.load(file)

        pd_class_names = sum([
            [class_name] * len(boxes_3d)
            for class_name, boxes_3d in prediction["boxes_3d"].items()
        ], [])

        pd_boxes_3d = torch.cat([
            torch.as_tensor(boxes_3d, dtype=torch.float)
            for boxes_3d in prediction["boxes_3d"].values()
        ], dim=0)

        pd_boxes_2d = torch.cat([
            torch.as_tensor(boxes_2d, dtype=torch.float)
            for boxes_2d in prediction["boxes_2d"].values()
        ], dim=0)

        pd_confidences = torch.cat([
            torch.as_tensor(confidences, dtype=torch.float)
            for confidences in prediction["confidences"].values()
        ], dim=0)

        annotation_filename = prediction_filename.replace(prediction_dirname, "annotations")

        with open(annotation_filename) as file:
            annotation = json.load(file)

        instance_ids = {
            class_name: list(masks.keys())
            for class_name, masks in annotation["masks"].items()
            if class_name in class_names
        }

        gt_class_names = sum([
            [class_name] *  len(instance_ids)
            for class_name, instance_ids in instance_ids.items()
        ], [])

        gt_boxes_3d = torch.cat([
            torch.as_tensor([
                annotation["boxes_3d"][class_name].get(instance_id, [[np.nan] * 3] * 8)
                for instance_id in instance_ids
            ], dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        gt_masks = torch.cat([
            torch.as_tensor([
                pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                for instance_id in instance_ids
            ], dtype=torch.float)
            for class_name, instance_ids in instance_ids.items()
        ], dim=0)

        gt_boxes_2d = torchvision.ops.masks_to_boxes(gt_masks).unflatten(-1, (2, 2))

        if not torch.all(torch.isfinite(gt_boxes_3d)): continue

        label_dirname = os.path.join("labels", os.path.basename(ckpt_dirname))
        label_filename = os.path.splitext(os.path.relpath(prediction_filename, root_dirname))[0]
        label_filename = os.path.join(root_dirname, label_dirname, f"{label_filename}.txt")

        os.makedirs(os.path.dirname(label_filename), exist_ok=True)

        save_prediction(
            filename=label_filename,
            class_names=pd_class_names,
            boxes_3d=pd_boxes_3d,
            boxes_2d=pd_boxes_2d,
            scores=pd_confidences,
        )

        label_dirname = os.path.join("labels", os.path.basename(ckpt_dirname))
        label_filename = os.path.splitext(os.path.relpath(annotation_filename, root_dirname))[0]
        label_filename = os.path.join(root_dirname, label_dirname, f"{label_filename}.txt")

        os.makedirs(os.path.dirname(label_filename), exist_ok=True)

        save_prediction(
            filename=label_filename,
            class_names=gt_class_names,
            boxes_3d=gt_boxes_3d,
            boxes_2d=gt_boxes_2d,
            scores=torch.ones(len(gt_class_names)),
        )


def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))

    with multiprocessing.Pool(args.num_workers) as pool:

        with tqdm.tqdm(total=len(sequences)) as progress_bar:

            for _ in pool.imap_unordered(functools.partial(
                convert_predictions,
                root_dirname=args.root_dirname,
                ckpt_dirname=args.ckpt_dirname,
                class_names=args.class_names,
            ), sequences):

                progress_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Prediction Converter for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--ckpt_dirname", type=str, default="ckpts/kitti_360/vsrd")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_workers", type=int, default=9)
    args = parser.parse_args()

    main(args)
