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
import numpy as np


OxtsPacket = collections.namedtuple(
    "OxtsPacket",
    (
        "lat", "lon", "alt",
        "roll", "pitch", "yaw",
        "vn", "ve",
        "vf", "vl", "vu",
        "ax", "ay", "az",
        "af", "al", "au",
        "wx", "wy", "wz",
        "wf", "wl", "wu",
        "posacc",
        "velacc",
        "navstat",
        "numsats",
        "posmode",
        "velmode",
        "orimode",
    ),
)


def compose_homogeneous_matrix(rotation_matrix, translation_vector):
    translation_vector = np.expand_dims(translation_vector, axis=-1)
    extrinsic_matrix = np.concatenate([rotation_matrix, translation_vector], axis=-1)
    homogeneous_vector = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    extrinsic_matrix = np.concatenate([extrinsic_matrix, homogeneous_vector], axis=0)
    return extrinsic_matrix


def compute_rotation_matrix_x(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation_matrix = np.array([
        [1.0, 0.0,  0.0],
        [0.0, cos, -sin],
        [0.0, sin,  cos],
    ])
    return rotation_matrix


def compute_rotation_matrix_y(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation_matrix = np.array([
        [ cos, 0.0, sin],
        [ 0.0, 1.0, 0.0],
        [-sin, 0.0, cos],
    ])
    return rotation_matrix


def compute_rotation_matrix_z(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation_matrix = np.array([
        [cos, -sin, 0.0],
        [sin,  cos, 0.0],
        [0.0,  0.0, 1.0],
    ])
    return rotation_matrix


def compute_imu2wld_matrix(oxts_packet, scale):

    # use the Euler angles to get the rotation matrix
    rotation_matrix_x = compute_rotation_matrix_x(oxts_packet.roll)
    rotation_matrix_y = compute_rotation_matrix_y(oxts_packet.pitch)
    rotation_matrix_z = compute_rotation_matrix_z(oxts_packet.yaw)
    rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x

    # earth radius (approx.) in meters
    earth_radius = 6378137.0

    # use a Mercator projection to get the translation vector
    translation_vector_x = earth_radius * scale * np.deg2rad(oxts_packet.lon)
    translation_vector_y = earth_radius * scale * np.log(np.tan(np.deg2rad(oxts_packet.lat) / 2.0 + np.pi / 4.0))
    translation_vector_z = oxts_packet.alt
    translation_vector = np.array([translation_vector_x, translation_vector_y, translation_vector_z])

    # combine the translation and rotation into a homogeneous transform
    imu2wld_matrix = compose_homogeneous_matrix(rotation_matrix, translation_vector)

    return imu2wld_matrix


def make_camera_params(sequence, root_dirname):

    # ================================================================
    # intrinsic matrix

    intrinsic_filename = os.path.join(root_dirname, sequence.rsplit("_", 3)[0], "calib_cam_to_cam.txt")

    with open(intrinsic_filename) as file:
        for line in file:
            name, *values = line.split()
            if name == "P_rect_02:":
                intrinsic_matrix = np.array(list(map(float, values))).reshape(3, 4)
                intrinsic_matrix, pixel_offset = np.split(intrinsic_matrix, [3], axis=-1)
                baseline = pixel_offset.squeeze(-1) / np.diag(intrinsic_matrix)
                translation_matrix = np.eye(4)
                translation_matrix[:-1, -1] = baseline
            if name == "R_rect_00:":
                cam2rect_matrix = np.array(list(map(float, values))).reshape(3, 3)
                cam2rect_matrix = compose_homogeneous_matrix(cam2rect_matrix, np.zeros(3))

    # ================================================================
    # extrinsic matrix

    imu2velo_filename = os.path.join(root_dirname, sequence.rsplit("_", 3)[0], "calib_imu_to_velo.txt")
    velo2cam_filename = os.path.join(root_dirname, sequence.rsplit("_", 3)[0], "calib_velo_to_cam.txt")

    with open(imu2velo_filename) as file:
        file.readline()
        _, *values = file.readline().split()
        rotation_matrix = np.array(list(map(float, values))).reshape(3, 3)
        _, *values = file.readline().split()
        translation_vector = np.array(list(map(float, values)))
        imu2velo_matrix = compose_homogeneous_matrix(rotation_matrix, translation_vector)

    with open(velo2cam_filename) as file:
        file.readline()
        _, *values = file.readline().split()
        rotation_matrix = np.array(list(map(float, values))).reshape(3, 3)
        _, *values = file.readline().split()
        translation_vector = np.array(list(map(float, values)))
        velo2cam_matrix = compose_homogeneous_matrix(rotation_matrix, translation_vector)

    imu2cam_matrix = cam2rect_matrix @ velo2cam_matrix @ imu2velo_matrix

    extrinsic_matrices = {}

    oxts_filenames = sorted(glob.glob(os.path.join(root_dirname, sequence.rsplit("_", 3)[0], sequence, "oxts", "data", "*.txt")))

    for oxts_filename in oxts_filenames:

        frame_index = int(os.path.splitext(os.path.basename(oxts_filename))[0])

        oxts_data = np.loadtxt(oxts_filename, delimiter=" ", skiprows=0)
        oxts_packet = OxtsPacket(*oxts_data)

        if not frame_index:
            scale = np.cos(np.deg2rad(oxts_packet.lat))
            source_imu2wld_matrix = compute_imu2wld_matrix(oxts_packet, scale)

        target_imu2wld_matrix = compute_imu2wld_matrix(oxts_packet, scale)

        extrinsic_matrix = imu2cam_matrix @ np.linalg.inv(target_imu2wld_matrix) @ source_imu2wld_matrix @ np.linalg.inv(imu2cam_matrix)
        extrinsic_matrices[frame_index] = extrinsic_matrix

    # ================================================================
    # image & segmentation mask

    image_filenames = sorted(glob.glob(os.path.join(root_dirname, sequence.rsplit("_", 3)[0], sequence, "image_02", "data", "*.png")))

    scale = None

    for image_filename in image_filenames:

        frame_index = int(os.path.splitext(os.path.basename(image_filename))[0])

        oxts_filename = os.path.join(root_dirname, sequence.rsplit("_", 3)[0], sequence, "oxts", "data", f"{frame_index:010}.txt")
        oxts_data = np.loadtxt(oxts_filename, delimiter=" ", skiprows=0)
        oxts_packet = OxtsPacket(*oxts_data)

        if not frame_index:
            scale = np.cos(np.deg2rad(oxts_packet.lat))
            source_imu2wld_matrix = compute_imu2wld_matrix(oxts_packet, scale)

        target_imu2wld_matrix = compute_imu2wld_matrix(oxts_packet, scale)

        extrinsic_matrix = imu2cam_matrix @ np.linalg.inv(target_imu2wld_matrix) @ source_imu2wld_matrix @ np.linalg.inv(imu2cam_matrix)

        if "image_02" in image_filename:
            extrinsic_matrix = translation_matrix @ extrinsic_matrix

        # FIXME: only consider monocular images
        assert "image_03" not in image_filename

        camera_param = dict(
            intrinsic_matrix=intrinsic_matrix.tolist(),
            extrinsic_matrix=extrinsic_matrix.tolist(),
        )

        camera_param_filename = image_filename.replace("image_02", "camera_params_02").replace(".png", ".json")
        os.makedirs(os.path.dirname(camera_param_filename), exist_ok=True)

        with open(camera_param_filename, "w") as file:
            json.dump(camera_param, file, indent=4, sort_keys=False)


def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "*", "*drive*")))))

    with multiprocessing.Pool(args.num_workers) as pool:

        with tqdm.tqdm(total=len(sequences)) as progress_bar:

            for _ in pool.imap_unordered(functools.partial(
                make_camera_params,
                root_dirname=args.root_dirname,
            ), sequences):

                progress_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Camera Parameter Maker for KITTI-Raw")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-Raw")
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    main(args)
