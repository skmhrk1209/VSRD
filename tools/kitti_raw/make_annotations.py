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
import cv2 as cv
import numpy as np
import scipy as sp
import pycocotools.mask

import mmdet.apis
import mmdet.datasets

import mmdet_custom

from tracker.byte_tracker import BYTETracker


class ByteTracker(object):

    def __init__(self, args, frame_rate):
        self.tracker = BYTETracker(args, frame_rate)

    def __call__(self, detection_result, image_size):

        tracklets = self.tracker.update(detection_result, image_size, image_size)

        boxes = np.asanyarray([tracklet.tlwh for tracklet in tracklets])
        scores = np.asanyarray([tracklet.score for tracklet in tracklets])
        instance_ids = np.asanyarray([tracklet.track_id for tracklet in tracklets])

        boxes[..., 2:] += boxes[..., :2]

        return boxes, scores, instance_ids


def tracking_by_detection(detector, tracker, image, class_names):

    assert len(class_names) == 1

    inference_results = collections.defaultdict(functools.partial(collections.defaultdict, functools.partial(collections.defaultdict, list)))

    (detection_results, segmenatation_results) = mmdet.apis.inference_detector(detector, image)

    for index, (detection_result, segmenatation_result) in enumerate(zip(detection_results, segmenatation_results)):

        if detector.CLASSES[index] not in class_names: continue

        boxes_2d, scores, instance_ids = tracker(detection_result, image.shape[:2])

        if not boxes_2d.size: continue

        # NOTE: workaround to select segmentation masks
        iou_matrix = torchvision.ops.box_iou(torch.from_numpy(boxes_2d), torch.from_numpy(detection_result[..., :-1])).numpy()
        _, indices = sp.optimize.linear_sum_assignment(iou_matrix, maximize=True)
        masks = np.asanyarray(segmenatation_result)[indices, ...]

        for mask, box_2d, score, instance_id in zip(masks, boxes_2d, scores, instance_ids):
            mask = np.asfortranarray(mask, dtype=np.uint8)
            mask = pycocotools.mask.encode(mask)
            mask = dict(mask, counts=mask["counts"].decode())
            inference_results["masks"][detector.CLASSES[index]][str(instance_id)] = mask
            inference_results["boxes_2d"][detector.CLASSES[index]][str(instance_id)]= box_2d.reshape(2, 2).tolist()
            inference_results["scores"][detector.CLASSES[index]][str(instance_id)] = score.item()

    return inference_results


def make_annotations(sequence, root_dirname, class_names, Detector, Tracker):

    detector = Detector()
    tracker = Tracker()

    image_filenames = sorted(glob.glob(os.path.join(root_dirname, sequence.rsplit("_", 3)[0], sequence, "image_02", "data", "*.png")))

    for image_filename in image_filenames:

        image = cv.imread(image_filename)
        inference_results = tracking_by_detection(detector, tracker, image, class_names)

        if not inference_results: continue

        camera_param_filename = image_filename.replace("image", "camera_params").replace(".png", ".json")
        with open(camera_param_filename) as file:
            camera_param = json.load(file)

        annotation = dict(**camera_param, **inference_results)

        annotation_filename = image_filename.replace("image", "annotations").replace(".png", ".json")
        os.makedirs(os.path.dirname(annotation_filename), exist_ok=True)

        with open(annotation_filename, "w") as file:
            json.dump(annotation, file, indent=4, sort_keys=False)


def main(args):

    sequences = list(map(os.path.basename, sorted(glob.glob(os.path.join(args.root_dirname, "*", "*drive*")))))

    Detector = functools.partial(mmdet.apis.init_detector, args.config, args.checkpoint, args.device)
    Tracker = functools.partial(ByteTracker, args=args, frame_rate=args.frame_rate)

    multiprocessing.set_start_method("spawn")

    with multiprocessing.Pool(args.num_workers) as pool:

        with tqdm.tqdm(total=len(sequences)) as progress_bar:

            for _ in pool.imap_unordered(functools.partial(
                make_annotations,
                root_dirname=args.root_dirname,
                class_names=args.class_names,
                Detector=Detector,
                Tracker=Tracker,
            ), sequences):

                progress_bar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VSRD: Annotation Maker for KITTI-Raw")
    parser.add_argument("--root_dirname", type=str, default="datasets/KITTI-Raw")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--config", type=str, default="../InternImage/detection/configs/coco/cascade_internimage_xl_fpn_3x_coco.py")
    parser.add_argument("--checkpoint", type=str, default="../InternImage/detection/checkpoints/cascade_internimage_xl_fpn_3x_coco.pth")
    parser.add_argument("--class_names", type=str, nargs="+", default=["car"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--mot20", action="store_true")
    parser.add_argument("--frame_rate", type=int, default=10)
    args = parser.parse_args()

    main(args)
