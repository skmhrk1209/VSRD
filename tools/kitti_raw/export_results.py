# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import os
import json
import glob
import shutil
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

    dimensions = torch.stack([widths, heights, lengths], dim=-1) # / 2.0

    orientations = torch.mean(torch.sub(
        boxes_3d[..., [1, 0, 4, 5], :],
        boxes_3d[..., [2, 3, 7, 6], :],
    ), dim=-2)

    orientations = nn.functional.normalize(orientations[..., [2, 0]], dim=-1)

    orientations = torch.atan2(*reversed(torch.unbind(orientations, dim=-1)))
    # orientations = rotation_matrix_y(*torch.unbind(orientations, dim=-1))

    return locations, dimensions, orientations


def export_result(filename, class_names, boxes_3d, boxes_2d, scores):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as file:

        for class_name, box_3d, box_2d, score in zip(class_names, boxes_3d, boxes_2d, scores):

            location, dimension, orientation = encode_box_3d(box_3d)

            # NOTE: KITTI-3D definition
            location[..., 1] += dimension[..., 1] / 2.0
            dimension = dimension[..., [1, 0, 2]]
            ray_orientation = torch.atan2(*reversed(torch.unbind(location[..., [2, 0]], dim=-1)))
            global_orientation = orientation - np.pi / 2.0
            local_orientation = global_orientation - ray_orientation

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


def export_results(sequence, root_dirname, ckpt_dirname, class_names):

    obj_train_filename = os.path.join(root_dirname, "splits", "obj_train.txt")
    with open(obj_train_filename) as file:
        obj_train_filenames = list(map(str.strip, file.readlines()))

    obj_valid_filename = os.path.join(root_dirname, "splits", "obj_valid.txt")
    with open(obj_valid_filename) as file:
        obj_valid_filenames = list(map(str.strip, file.readlines()))

    raw_train_filename = os.path.join(root_dirname, "splits", "raw_train.txt")
    with open(raw_train_filename) as file:
        raw_train_filenames = list(map(str.strip, file.readlines()))

    obj_to_raw_filename = os.path.join(root_dirname, "splits", "obj_to_raw.txt")
    with open(obj_to_raw_filename) as file:
        obj_to_raw_filenames = dict(map(str.split, file.readlines()))
        raw_to_obj_filenames = dict(map(reversed, obj_to_raw_filenames.items()))

    prediction_dirname = os.path.join("predictions_02", os.path.basename(ckpt_dirname))
    prediction_filenames = sorted(glob.glob(os.path.join(root_dirname, sequence.rsplit("_", 3)[0], sequence, prediction_dirname, "data", "*.json")))

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

        image_filename = prediction_filename.replace(prediction_dirname, "image_02").replace(".json", ".png")

        if os.path.relpath(image_filename, root_dirname) in raw_train_filenames:

            result_dirname = os.path.join("results", os.path.basename(ckpt_dirname), "raw_train")
            result_filename = os.path.splitext(os.path.relpath(prediction_filename, root_dirname))[0]
            result_filename = os.path.join(root_dirname, result_dirname, f"{result_filename}.txt")

            os.makedirs(os.path.dirname(result_filename), exist_ok=True)
            export_result(
                filename=result_filename,
                class_names=pd_class_names,
                boxes_3d=pd_boxes_3d,
                boxes_2d=pd_boxes_2d,
                scores=pd_confidences,
            )

        if os.path.relpath(image_filename, root_dirname) in obj_train_filenames:

            result_dirname = os.path.join("results", os.path.basename(ckpt_dirname), "obj_train")
            result_filename = os.path.splitext(os.path.relpath(prediction_filename, root_dirname))[0]
            result_filename = os.path.join(root_dirname, result_dirname, f"{result_filename}.txt")

            os.makedirs(os.path.dirname(result_filename), exist_ok=True)
            export_result(
                filename=result_filename,
                class_names=pd_class_names,
                boxes_3d=pd_boxes_3d,
                boxes_2d=pd_boxes_2d,
                scores=pd_confidences,
            )

            annotation_filename = prediction_filename.replace(prediction_dirname, "annotations_02")

            result_dirname = os.path.join("results", os.path.basename(ckpt_dirname), "obj_train")
            result_filename = os.path.splitext(os.path.relpath(annotation_filename, root_dirname))[0]
            result_filename = os.path.join(root_dirname, result_dirname, f"{result_filename}.txt")

            label_filename = raw_to_obj_filenames[os.path.relpath(image_filename, root_dirname)]
            label_filename = os.path.join(root_dirname.replace("Raw", "Object"), label_filename.replace("image", "label").replace(".png", ".txt"))

            os.makedirs(os.path.dirname(result_filename), exist_ok=True)
            shutil.copy(label_filename, result_filename)


def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "*", "*drive*")))))

    with multiprocessing.Pool(args.num_workers) as pool:

        with tqdm.tqdm(total=len(sequences)) as progress_bar:

            for _ in pool.imap_unordered(functools.partial(
                export_results,
                root_dirname=args.root_dirname,
                ckpt_dirname=args.ckpt_dirname,
                class_names=args.class_names,
            ), sequences):

                progress_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Result Exporter for KITTI-Raw")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-Raw")
    parser.add_argument("--ckpt_dirname", type=str, default="ckpts/kitti_raw/vsrd")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    main(args)
