import os
import json
import glob
import operator
import argparse
import functools
import itertools
import collections
import multiprocessing

import tqdm
import numpy as np
import pycocotools.mask


def sample_annotations(
    sequence,
    root_dirname,
    class_names,
    num_instance_ratio,
    num_source_frames,
    min_mask_area,
    min_box_size,
):
    image_filenames = sorted(glob.glob(os.path.join(root_dirname, "data_2d_raw", sequence, "image_00", "data_rect", "*.png")))

    frame_indices = [
        int(os.path.splitext(os.path.basename(image_filename))[0])
        for image_filename in image_filenames
    ]

    min_frame_index = min(frame_indices)
    max_frame_index = max(frame_indices)

    grouped_image_filenames = collections.defaultdict(list)

    for target_image_filename in image_filenames:

        def sample_source_frames(target_annotation_filename):

            if not os.path.exists(target_annotation_filename): return [], []

            with open(target_annotation_filename) as file:
                target_annotation = json.load(file)

            def check_mask(mask):
                mask_area = np.sum(mask)
                indices = np.where(mask)
                min_indices = list(map(min, indices))
                max_indices = list(map(max, indices))
                box_size = np.min(np.subtract(max_indices, min_indices))
                return (mask_area >= min_mask_area) & (box_size >= min_box_size)

            target_instance_ids = sum([
                [
                    instance_id
                    for instance_id, mask in masks.items()
                    if check_mask(pycocotools.mask.decode(mask))
                ]
                for class_name, masks
                in target_annotation["masks"].items()
                if class_name in class_names
            ], [])

            if not target_instance_ids: return [], []

            source_relative_indices = []

            target_frame_index = int(os.path.splitext(os.path.basename(target_annotation_filename))[0])

            for source_relative_index in itertools.count(1):

                source_frame_index = target_frame_index + source_relative_index

                if max_frame_index < source_frame_index: break

                source_annotation_filename = os.path.join(os.path.dirname(target_annotation_filename), f"{source_frame_index:010}.json")

                if not os.path.exists(source_annotation_filename): continue

                with open(source_annotation_filename) as file:
                    source_annotation = json.load(file)

                source_instance_ids = sum([
                    [
                        instance_id
                        for instance_id, mask in masks.items()
                        if check_mask(pycocotools.mask.decode(mask))
                    ]
                    for class_name, masks
                    in source_annotation["masks"].items()
                    if class_name in class_names
                ], [])

                if len(set(target_instance_ids) & set(source_instance_ids)) / len(target_instance_ids) < num_instance_ratio: break

                source_relative_indices.append(source_relative_index)

            for source_relative_index in itertools.count(1):

                source_frame_index = target_frame_index - source_relative_index

                if source_frame_index < min_frame_index: break

                source_annotation_filename = os.path.join(os.path.dirname(target_annotation_filename), f"{source_frame_index:010}.json")

                if not os.path.exists(source_annotation_filename): continue

                with open(source_annotation_filename) as file:
                    source_annotation = json.load(file)

                source_instance_ids = sum([
                    [
                        instance_id
                        for instance_id, mask in masks.items()
                        if check_mask(pycocotools.mask.decode(mask))
                    ]
                    for class_name, masks
                    in source_annotation["masks"].items()
                    if class_name in class_names
                ], [])

                if len(set(target_instance_ids) & set(source_instance_ids)) / len(target_instance_ids) < num_instance_ratio: break

                source_relative_indices.append(-source_relative_index)

            return sorted(target_instance_ids), sorted(source_relative_indices)

        target_annotation_filename = target_image_filename.replace("data_2d_raw", "annotations").replace(".png", ".json")
        target_instance_ids, source_relative_indices = sample_source_frames(target_annotation_filename)

        if len(source_relative_indices) >= num_source_frames:
            grouped_image_filenames[tuple(target_instance_ids)].append((target_image_filename, source_relative_indices))

    grouped_image_filename = os.path.join(
        root_dirname,
        f"filenames",
        f"R{num_instance_ratio * 100.0:.0f}-"
        f"N{num_source_frames}-"
        f"M{min_mask_area}-"
        f"B{min_box_size}",
        sequence,
        "grouped_image_filenames.txt"
    )

    sampled_image_filename = os.path.join(
        root_dirname,
        f"filenames",
        f"R{num_instance_ratio * 100.0:.0f}-"
        f"N{num_source_frames}-"
        f"M{min_mask_area}-"
        f"B{min_box_size}",
        sequence,
        "sampled_image_filenames.txt"
    )

    os.makedirs(os.path.dirname(grouped_image_filename), exist_ok=True)
    os.makedirs(os.path.dirname(sampled_image_filename), exist_ok=True)

    with open(grouped_image_filename, "w") as grouped_image_file:

        with open(sampled_image_filename, "w") as sampled_image_file:

            for target_instance_ids, grouped_image_filenames in grouped_image_filenames.items():

                grouped_image_filenames = sorted(grouped_image_filenames, key=lambda item: int(os.path.splitext(os.path.basename(item[0]))[0]))
                target_image_filename, source_relative_indices = grouped_image_filenames[len(grouped_image_filenames) // 2]

                grouped_image_file.write(f"{','.join(target_instance_ids)} {','.join(map(operator.itemgetter(0), grouped_image_filenames))}\n")
                sampled_image_file.write(f"{','.join(target_instance_ids)} {target_image_filename} {','.join(map(str, source_relative_indices))}\n")

def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "data_2d_raw", "*")))))

    with multiprocessing.Pool(args.num_workers) as pool:

        with tqdm.tqdm(total=len(sequences)) as progress_bar:

            for _ in pool.imap_unordered(functools.partial(
                sample_annotations,
                root_dirname=args.root_dirname,
                class_names=args.class_names,
                num_instance_ratio=args.num_instance_ratio,
                num_source_frames=args.num_source_frames,
                min_mask_area=args.min_mask_area,
                min_box_size=args.min_box_size,
            ), sequences):

                progress_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Annotation Sampler for KITTI-360")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-360")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--num_instance_ratio", type=float, default=0.5)
    parser.add_argument("--num_source_frames", type=int, default=16)
    parser.add_argument("--min_mask_area", type=int, default=128)
    parser.add_argument("--min_box_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=9)
    args = parser.parse_args()

    main(args)
