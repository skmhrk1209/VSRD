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
import scipy as sp
import pycocotools.mask

import vsrd


LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def make_predictions(
    sequence,
    root_dirname,
    ckpt_dirname,
    ckpt_filename,
    split_dirname,
    class_names,
):
    group_filename = os.path.join(root_dirname, "filenames", split_dirname, sequence, "grouped_image_filenames.txt")

    with open(group_filename) as file:
        grouped_image_filenames = {
            tuple(map(int, line.split(" ")[0].split(","))): line.split(" ")[1].split(",")
            for line in map(str.strip, file)
        }

    sample_filename = os.path.join(root_dirname, "filenames", split_dirname, sequence, "sampled_image_filenames.txt")

    with open(sample_filename) as file:
        sampled_image_filenames = {
            tuple(map(int, line.split(" ")[0].split(","))): line.split(" ")[1]
            for line in map(str.strip, file)
        }

    for instance_ids, grouped_image_filenames in grouped_image_filenames.items():

        target_image_filename = sampled_image_filenames[instance_ids]

        target_image_dirname = os.path.splitext(os.path.relpath(target_image_filename, root_dirname))[0]
        target_ckpt_filename = os.path.join(ckpt_dirname, sequence, target_image_dirname, ckpt_filename)

        if not os.path.exists(target_ckpt_filename):
            print(f"[{target_ckpt_filename}] Does not exist!")
            continue

        target_checkpoint = torch.load(target_ckpt_filename, map_location="cpu")

        model = vsrd.models.BoxParameters3D(*torch.tensor(target_checkpoint["models"]["detector"]["embeddings"]).shape)
        model.load_state_dict(target_checkpoint["models"]["detector"])

        world_boxes_3d, = model()["boxes_3d"]
        world_boxes_3d = nn.functional.pad(world_boxes_3d, (0, 1), mode="constant", value=1.0)

        target_annotation_filename = target_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")

        with open(target_annotation_filename) as file:
            target_annotation = json.load(file)

        target_extrinsic_matrix = torch.tensor(target_annotation["extrinsic_matrix"])
        inverse_target_extrinsic_matrix = torch.linalg.inv(target_extrinsic_matrix)

        x_axis, y_axis, _ = target_extrinsic_matrix[..., :3, :3]
        rectification_angle = (
            torch.acos(torch.dot(torch.round(y_axis), y_axis)) *
            torch.sign(torch.dot(torch.cross(torch.round(y_axis), y_axis), x_axis))
        )
        rectification_matrix = vsrd.operations.rotation_matrix_x(rectification_angle)

        target_instance_ids = torch.cat([
            torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
            for class_name, masks in target_annotation["masks"].items()
            if class_name in class_names
        ], dim=0)

        accumulated_iou_matrix = torch.zeros(len(world_boxes_3d), len(target_instance_ids))
        accumulated_cnt_matrix = torch.zeros(len(world_boxes_3d), len(target_instance_ids))

        callbacks = []

        for source_image_filename in grouped_image_filenames:

            source_annotation_filename = source_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")

            with open(source_annotation_filename) as file:
                source_annotation = json.load(file)

            source_intrinsic_matrix = torch.tensor(source_annotation["intrinsic_matrix"])
            source_extrinsic_matrix = torch.tensor(source_annotation["extrinsic_matrix"])

            source_extrinsic_matrix = (
                source_extrinsic_matrix @
                inverse_target_extrinsic_matrix @
                vsrd.operations.expand_to_4x4(rectification_matrix.T)
            )

            source_pd_boxes_3d = world_boxes_3d @ source_extrinsic_matrix.T
            source_pd_boxes_3d = source_pd_boxes_3d[..., :-1] / source_pd_boxes_3d[..., -1:]

            source_pd_boxes_2d = torch.stack([
                vsrd.operations.project_box_3d(
                    box_3d=source_pd_box_3d,
                    line_indices=LINE_INDICES,
                    intrinsic_matrix=source_intrinsic_matrix,
                )
                for source_pd_box_3d in source_pd_boxes_3d
            ], dim=0)

            source_gt_masks = torch.cat([
                torch.as_tensor(list(map(pycocotools.mask.decode, masks.values())), dtype=torch.float)
                for class_name, masks in source_annotation["masks"].items()
                if class_name in class_names
            ], dim=0)

            source_gt_masks = vsrd.transforms.MaskRefiner()(source_gt_masks)["masks"]
            source_gt_boxes_2d = torchvision.ops.masks_to_boxes(source_gt_masks.bool()).unflatten(-1, (2, 2))

            source_pd_boxes_2d = torchvision.ops.clip_boxes_to_image(
                boxes=source_pd_boxes_2d.flatten(-2, -1),
                size=source_gt_masks.shape[-2:],
            ).unflatten(-1, (2, 2))

            source_iou_matrix = torch.nan_to_num(torchvision.ops.box_iou(
                boxes1=source_pd_boxes_2d.flatten(-2, -1),
                boxes2=source_gt_boxes_2d.flatten(-2, -1),
            ))

            source_instance_ids = torch.cat([
                torch.as_tensor(list(map(int, masks.keys())), dtype=torch.long)
                for class_name, masks in source_annotation["masks"].items()
                if class_name in class_names
            ], dim=0)

            target_gt_indices = source_instance_ids.new_tensor([
                target_instance_ids.tolist().index(source_instance_id.item())
                if source_instance_id in target_instance_ids else -1
                for source_instance_id in source_instance_ids
            ])

            accumulated_iou_matrix[
                ...,
                target_gt_indices[target_gt_indices >= 0]
            ] += source_iou_matrix[..., target_gt_indices >= 0]

            accumulated_cnt_matrix[
                ...,
                target_gt_indices[target_gt_indices >= 0]
            ] += 1

            def save_prediction(filename, boxes_3d, boxes_2d, confidences):

                prediction = dict(
                    boxes_3d=dict(car=boxes_3d.tolist()),
                    boxes_2d=dict(car=boxes_2d.tolist()),
                    confidences=dict(car=confidences.tolist()),
                )

                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as file:
                    json.dump(prediction, file, indent=4, sort_keys=False)

            source_prediction_dirname = os.path.join("predictions", os.path.basename(ckpt_dirname))
            source_prediction_filename = source_annotation_filename.replace("annotations", source_prediction_dirname)

            callbacks.append(functools.partial(
                save_prediction,
                filename=source_prediction_filename,
                boxes_3d=source_pd_boxes_3d,
                boxes_2d=source_pd_boxes_2d,
            ))

        averaged_iou_matrix = accumulated_iou_matrix / accumulated_cnt_matrix
        matched_pd_indices, matched_gt_indices = vsrd.utils.torch_function(sp.optimize.linear_sum_assignment)(averaged_iou_matrix, maximize=True)

        confidences = averaged_iou_matrix[matched_pd_indices, matched_gt_indices]

        for callback in callbacks:
            callback(confidences=confidences)


def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))

    with multiprocessing.Pool(args.num_workers) as pool:

        with tqdm.tqdm(total=len(sequences)) as progress_bar:

            for _ in pool.imap_unordered(functools.partial(
                make_predictions,
                root_dirname=args.root_dirname,
                ckpt_dirname=args.ckpt_dirname,
                ckpt_filename=args.ckpt_filename,
                split_dirname=args.split_dirname,
                class_names=args.class_names,
            ), sequences):

                progress_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Prediction Maker for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--ckpt_dirname", type=str, default="ckpts/kitti_360/vsrd")
    parser.add_argument("--ckpt_filename", type=str, default="step_2999.pt")
    parser.add_argument("--split_dirname", type=str, default="R50-N16-M128-B16")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_workers", type=int, default=9)
    args = parser.parse_args()

    main(args)
