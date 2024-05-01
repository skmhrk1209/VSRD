import os
import json
import glob
import argparse
import functools
import collections
import multiprocessing
import xml.etree.ElementTree

import tqdm
import cv2 as cv
import numpy as np
import pycocotools.mask

from kitti360scripts.helpers import labels


def make_annotations(sequence, root_dirname):

    # ================================================================
    # intrinsic matrix

    intrinsic_filename = os.path.join(root_dirname, "calibration", "perspective.txt")

    with open(intrinsic_filename) as file:
        for line in file:
            name, *values = line.split()
            if name == "P_rect_01:":
                intrinsic_matrix = np.array(list(map(float, values))).reshape(3, 4)
                intrinsic_matrix, pixel_offset = np.split(intrinsic_matrix, [3], axis=-1)
                baseline = pixel_offset.squeeze(-1) / np.diag(intrinsic_matrix)
                translation_matrix = np.eye(4)
                translation_matrix[:-1, -1] = baseline
                break

    # ================================================================
    # extrinsic matrix

    extrinsic_filename = os.path.join(root_dirname, "data_poses", sequence, "cam0_to_world.txt")

    extrinsic_matrices = {}
    with open(extrinsic_filename) as file:
        for line in file:
            frame_index, *values = line.split()
            frame_index = int(frame_index)
            cam2wld_matrix = np.array(list(map(float, values))).reshape(4, 4)
            wld2cam_matrix = np.linalg.inv(cam2wld_matrix)
            extrinsic_matrices[frame_index] = wld2cam_matrix

    # ================================================================
    # 3D bounding box

    box_3d_filename = os.path.join(root_dirname, "data_3d_bboxes", "train", f"{sequence}.xml")
    tree = xml.etree.ElementTree.parse(box_3d_filename)

    wld_boxes_3d = collections.defaultdict(dict)
    for child in tree.getroot():

        kitti_semantic_id = int(child.find("semanticId").text)
        class_instance_id = int(child.find("instanceId").text)

        semantic_id = labels.kittiId2label[kitti_semantic_id].id
        instance_id = semantic_id * 1000 + class_instance_id

        data = child.find("transform").find("data").text.split()
        obj2wld_matrix = np.array(list(map(float, data))).reshape(4, 4)

        data = child.find("vertices").find("data").text.split()
        obj_box_3d = np.array(list(map(float, data))).reshape(8, 3)

        # NOTE: convert from the KITTI-360 "annotation" format to the KITTI-360 "evaluation" format
        # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/prepare_train_val_windows.py#L133
        # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L552
        obj_box_3d = obj_box_3d[[0, 2, 7, 5, 1, 3, 6, 4], ...]

        obj_box_3d = np.pad(obj_box_3d, ((0, 0), (0, 1)), constant_values=1.0)
        wld_box_3d = obj_box_3d @ obj2wld_matrix.T

        frame_index = int(child.find("timestamp").text)
        wld_boxes_3d[frame_index][instance_id] = wld_box_3d

    # ================================================================
    # image & instance mask

    image_filenames = sorted(glob.glob(os.path.join(root_dirname, "data_2d_raw", sequence, "**", "*.png"), recursive=True))

    for image_filename in image_filenames:

        frame_index = int(os.path.splitext(os.path.basename(image_filename))[0])
        instance_filename = image_filename.replace("data_2d_raw", "data_2d_semantics/train").replace("data_rect", "instance")

        if frame_index not in extrinsic_matrices: continue
        if not os.path.exists(instance_filename): continue

        extrinsic_matrix = extrinsic_matrices[frame_index]

        if "image_01" in image_filename:
            extrinsic_matrix = translation_matrix @ extrinsic_matrix

        annotation = collections.defaultdict(
            functools.partial(collections.defaultdict, dict),
            intrinsic_matrix=intrinsic_matrix.tolist(),
            extrinsic_matrix=extrinsic_matrix.tolist(),
        )

        instance_map = cv.imread(instance_filename, cv.IMREAD_ANYDEPTH)

        for instance_id in np.unique(instance_map).tolist():

            semantic_id = instance_id // 1000
            class_name = labels.id2label[semantic_id].name

            mask = instance_map == instance_id
            mask = np.asfortranarray(mask, dtype=np.uint8)
            mask = pycocotools.mask.encode(mask)
            mask = dict(mask, counts=mask["counts"].decode())
            annotation["masks"][class_name][instance_id] = mask

            annotation["boxes_3d"][class_name]

            if instance_id in wld_boxes_3d[frame_index]:
                wld_box_3d = wld_boxes_3d[frame_index][instance_id]

            elif instance_id in wld_boxes_3d[-1]:
                wld_box_3d = wld_boxes_3d[-1][instance_id]

            else: continue

            cam_box_3d = wld_box_3d @ extrinsic_matrix.T
            cam_box_3d = cam_box_3d[..., :-1] / cam_box_3d[..., -1:]

            annotation["boxes_3d"][class_name][instance_id] = cam_box_3d.tolist()

        annotation_filename = image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
        os.makedirs(os.path.dirname(annotation_filename), exist_ok=True)

        with open(annotation_filename, "w") as file:
            json.dump(annotation, file, indent=4, sort_keys=False)


def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))

    with multiprocessing.Pool(args.num_workers) as pool:

        with tqdm.tqdm(total=len(sequences)) as progress_bar:

            for _ in pool.imap_unordered(functools.partial(
                make_annotations,
                root_dirname=args.root_dirname,
            ), sequences):

                progress_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Annotation Maker for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--num_workers", type=int, default=9)
    args = parser.parse_args()

    main(args)
