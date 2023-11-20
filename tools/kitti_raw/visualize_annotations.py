# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import os
import json
import glob
import argparse
import functools
import collections
import multiprocessing

import tqdm
import torch
import torchvision
import numpy as np
import cv2 as cv
import pycocotools.mask


def visualize_annotations(sequence, root_dirname, out_dirname, class_names, frame_rate):

    video_writer = None

    color_palette = collections.defaultdict(functools.partial(np.random.randint, 0, 1 << 8, (3,)))

    image_filenames = sorted(glob.glob(os.path.join(root_dirname, sequence.rsplit("_", 3)[0], sequence, "image_02", "data", "*.png")))

    for image_filename in image_filenames:

        annotation_filename = image_filename.replace("image", "annotations").replace(".png", ".json")

        if not os.path.exists(annotation_filename): continue

        with open(annotation_filename) as file:
            annotation = json.load(file)

        if "masks" not in annotation: continue

        masks = torch.cat([
            torch.as_tensor(list(map(pycocotools.mask.decode, masks.values())), dtype=torch.float)
            for class_name, masks in annotation["masks"].items()
            if class_name in class_names
        ], dim=0)

        boxes_2d = torch.cat([
            torch.as_tensor(list(boxes_2d.values()), dtype=torch.float)
            for class_name, boxes_2d in annotation["boxes_2d"].items()
            if class_name in class_names
        ], dim=0)

        instance_ids = sum([
            list(map(int, masks.keys()))
            for class_name, masks in annotation["masks"].items()
            if class_name in class_names
        ], [])

        colors = [
            tuple((color_palette[instance_id] / np.max(color_palette[instance_id]) * (1 << 8) - 1).astype(int).tolist())
            for instance_id in instance_ids
        ]

        image = torchvision.io.read_image(image_filename)

        image = torchvision.utils.draw_segmentation_masks(
            image=image,
            masks=masks > 0.5,
            alpha=0.5,
            colors=colors,
        )

        image = torchvision.utils.draw_bounding_boxes(
            image=image,
            boxes=boxes_2d.flatten(-2, -1),
            colors=colors,
        )

        if not video_writer:
            video_filename = os.path.join(out_dirname, f"{sequence}.mp4")
            os.makedirs(os.path.dirname(video_filename), exist_ok=True)
            video_codec = cv.VideoWriter_fourcc(*"mp4v")
            video_writer = cv.VideoWriter(video_filename, video_codec, frame_rate, image.shape[:2][::-1])

        video_writer.write(image.permute(1, 2, 0).numpy())

        frame_index = int(os.path.splitext(os.path.basename(image_filename))[0])
        image_filename = os.path.join(out_dirname, sequence, f"{frame_index:010}.png")
        os.makedirs(os.path.dirname(image_filename), exist_ok=True)
        torchvision.io.write_png(image, image_filename)

    video_writer.release()


def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "*", "*drive*")))))

    with multiprocessing.Pool(args.num_workers) as pool:

        with tqdm.tqdm(total=len(sequences)) as progress_bar:

            for _ in pool.imap_unordered(functools.partial(
                visualize_annotations,
                root_dirname=args.root_dirname,
                out_dirname=args.out_dirname,
                class_names=args.class_names,
                frame_rate=args.frame_rate,
            ), sequences):

                progress_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Annotation Visualizer for KITTI-Raw")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-Raw")
    parser.add_argument("--out_dirname", type=str, default="videos/kitti_raw/annotations")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--frame_rate", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    main(args)
