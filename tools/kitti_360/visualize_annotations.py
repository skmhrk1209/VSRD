# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import os
import json
import glob
import argparse
import functools
import multiprocessing

import tqdm
import torch
import torchvision
import cv2 as cv
import pycocotools.mask

import vsrd


LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def visualize_annotations(sequence, root_dirname, out_dirname, class_names, frame_rate):

    video_writer = None

    image_filenames = sorted(glob.glob(os.path.join(root_dirname, "data_2d_raw", sequence, "image_00", "data_rect", "*.png")))

    for image_filename in image_filenames:

        annotation_filename = image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")

        if not os.path.exists(annotation_filename): continue

        with open(annotation_filename) as file:
            annotation = json.load(file)

        masks = torch.cat([
            torch.as_tensor(list(map(pycocotools.mask.decode, masks.values())), dtype=torch.float)
            for class_name, masks in annotation["masks"].items()
            if class_name in class_names
        ], dim=0)

        boxes_3d = torch.cat([
            torch.as_tensor(list(boxes_3d.values()), dtype=torch.float)
            for class_name, boxes_3d in annotation["boxes_3d"].items()
            if class_name in class_names
        ], dim=0)

        intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])

        image = torchvision.io.read_image(image_filename)

        image = vsrd.visualization.draw_masks(image, masks)

        image = vsrd.visualization.draw_boxes_3d(
            image=image,
            boxes_3d=boxes_3d,
            line_indices=LINE_INDICES + [[0, 5], [1, 4]],
            intrinsic_matrix=intrinsic_matrix,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv.LINE_AA,
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

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))

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

    parser = argparse.ArgumentParser(description="VSRD: Annotation Visualizer for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--out_dirname", type=str, default="videos/kitti_360/annotations")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--frame_rate", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=9)
    args = parser.parse_args()

    main(args)
