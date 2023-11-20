import os
import json
import random
import operator
import functools
import itertools
import multiprocessing

import torch
import torchvision
import numpy as np
import skimage
import pycocotools.mask

from .. import operations


class KITTIRawDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        filenames,
        class_names,
        num_workers=4,
        num_source_frames=2,
        target_transforms=[],
        source_transforms=[],
        rectification=True,
    ):
        super().__init__()

        self.image_filenames = []
        self.image_blacklist = set()

        for filename in filenames:
            with open(filename) as file:
                for line in file:
                    _, target_image_filename, source_relative_indices = line.strip().split(" ")
                    source_relative_indices = list(map(int, source_relative_indices.split(",")))
                    self.image_filenames.append((target_image_filename, source_relative_indices))

        self.filenames = filenames
        self.class_names = class_names
        self.num_workers = num_workers
        self.num_source_frames = num_source_frames
        self.target_transforms = target_transforms
        self.source_transforms = source_transforms
        self.rectification = rectification

    @staticmethod
    def get_root_dirname(image_filename):
        root_dirname = functools.reduce(lambda x, f: f(x), [os.path.dirname] * 5, image_filename)
        return root_dirname

    @staticmethod
    def get_sequence_dirname(image_filename):
        sequence_dirname = functools.reduce(lambda x, f: f(x), [os.path.dirname] * 3, image_filename)
        return sequence_dirname

    @staticmethod
    def get_annotation_filename(image_filename):
        annotation_filename = (
            image_filename
            .replace("image", "annotations")
            .replace(".png", ".json")
        )
        return annotation_filename

    @staticmethod
    def get_image_filename(image_filename, relative_index=0):
        frame_index = int(os.path.splitext(os.path.basename(image_filename))[0])
        image_filename = os.path.join(
            os.path.dirname(image_filename),
            f"{frame_index + relative_index:010}.png",
        )
        return image_filename

    @staticmethod
    def read_image(image_filename):
        image = skimage.io.imread(image_filename)
        image = torchvision.transforms.functional.to_tensor(image)
        return image

    def read_annotation(self, annotation_filename):

        with open(annotation_filename) as file:
            annotation = json.load(file)

        intrinsic_matrix = torch.as_tensor(annotation["intrinsic_matrix"])
        extrinsic_matrix = torch.as_tensor(annotation["extrinsic_matrix"])

        instance_ids = {
            class_name: list(masks.keys())
            for class_name, masks in annotation["masks"].items()
            if class_name in self.class_names
        }

        if instance_ids:

            masks = torch.cat([
                torch.as_tensor(np.stack([
                    pycocotools.mask.decode(annotation["masks"][class_name][instance_id])
                    for instance_id in instance_ids
                ]), dtype=torch.float)
                for class_name, instance_ids in instance_ids.items()
            ], dim=0)

            labels = torch.cat([
                torch.as_tensor([self.class_names.index(class_name)] * len(instance_ids), dtype=torch.long)
                for class_name, instance_ids in instance_ids.items()
            ], dim=0)

            boxes_3d = torch.cat([
                torch.as_tensor([
                    [[np.nan] * 3] * 8
                    for instance_id in instance_ids
                ], dtype=torch.float)
                for class_name, instance_ids in instance_ids.items()
            ], dim=0)

            instance_ids = torch.cat([
                torch.as_tensor(list(map(int, instance_ids)), dtype=torch.long)
                for instance_ids in instance_ids.values()
            ], dim=0)

            return dict(
                masks=masks,
                labels=labels,
                boxes_3d=boxes_3d,
                instance_ids=instance_ids,
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
            )

        else:

            return dict(
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
            )

    def __len__(self):
        return len(self.image_filenames)

    def getitem(self, image_filename, transforms=[]):

        annotation_filename = __class__.get_annotation_filename(image_filename)

        image = __class__.read_image(image_filename)
        annotation = self.read_annotation(annotation_filename)

        list(itertools.starmap(
            annotation.setdefault,
            dict(
                masks=torch.empty(0, *image.shape[-2:], dtype=torch.float),
                labels=torch.empty(0, dtype=torch.long),
                boxes_3d=torch.empty(0, 8, 3, dtype=torch.float),
                instance_ids=torch.empty(0, dtype=torch.long),
            ).items(),
        ))

        inputs = dict(
            annotation,
            image=image,
            filename=image_filename,
        )

        for transform in transforms:
            inputs = transform(**inputs)

        return inputs

    def __getitem__(self, index):

        target_image_filename, source_relative_indices = self.image_filenames[index]

        if target_image_filename in self.image_blacklist:
            return random.choice(self)

        target_inputs = self.getitem(
            image_filename=target_image_filename,
            transforms=self.target_transforms,
        )

        if not len(target_inputs["masks"]):
            print(f"[{target_image_filename}] No instances. Added to the blacklist.")
            self.image_blacklist.add(target_image_filename)
            return random.choice(self)

        multi_inputs = {0: target_inputs}

        source_relative_indices = [
            source_relative_indices[len(source_relative_indices) // 2]
            for source_relative_indices
            in np.array_split(source_relative_indices, self.num_source_frames)
            if source_relative_indices.size
        ]

        with multiprocessing.Pool(self.num_workers) as pool:
            multi_inputs.update(dict(zip(
                source_relative_indices,
                pool.imap(
                    functools.partial(self.getitem, transforms=self.source_transforms),
                    [
                        __class__.get_image_filename(
                            image_filename=target_image_filename,
                            relative_index=source_relative_index,
                        )
                        for source_relative_index in source_relative_indices
                    ],
                ),
            )))

        multi_inputs = dict(sorted(multi_inputs.items(), key=operator.itemgetter(0)))

        if self.rectification:

            target_extrinsic_matrix = target_inputs["extrinsic_matrix"]
            inverse_target_extrinsic_matrix = torch.linalg.inv(target_extrinsic_matrix)

            x_axis, y_axis, _ = target_extrinsic_matrix[..., :3, :3]
            rectification_angle = (
                torch.acos(torch.dot(torch.round(y_axis), y_axis)) *
                torch.sign(torch.dot(torch.cross(torch.round(y_axis), y_axis), x_axis))
            )
            rectification_matrix = operations.rotation_matrix_x(rectification_angle)

            for source_inputs in multi_inputs.values():

                source_extrinsic_matrix = source_inputs["extrinsic_matrix"]

                source_extrinsic_matrix = (
                    source_extrinsic_matrix @
                    inverse_target_extrinsic_matrix @
                    operations.expand_to_4x4(rectification_matrix.T)
                )

                source_inputs.update(
                    extrinsic_matrix=source_extrinsic_matrix,
                    rectification_matrix=rectification_matrix,
                )

        for transforms in [self.target_transforms, self.source_transforms]:
            for transform in transforms:
                if hasattr(transform, "update_params"):
                    transform.update_params()

        return multi_inputs
