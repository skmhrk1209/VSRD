import os
import re
import json
import random
import shutil
import logging
import datetime
import argparse
import operator
import functools
import multiprocessing

import torch
import torch.nn as nn
import torchvision
import cv2 as cv
import numpy as np
import scipy as sp
import inflection

import torch.utils.tensorboard

import vsrd


LINE_INDICES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def main(args):

    # ================================================================
    # configuration

    config = vsrd.configuration.Configurator.load(args.config)
    config = vsrd.utils.Dict.apply(config)
    config.update(vars(args))

    # ================================================================
    # distributed data parallelism

    if config.launcher == "slurm":
        # NOTE: we must specify `MASTER_ADDR` and `MASTER_PORT` by the environment variables
        vsrd.distributed.init_process_group(backend=config.distributed.backend, port=config.port)
    if config.launcher == "torch":
        torch.distributed.init_process_group(backend=config.distributed.backend)

    device_id = vsrd.distributed.get_device_id(config.distributed.num_devices_per_process, args.device_id)

    for rank in range(torch.distributed.get_world_size()):
        with vsrd.distributed.barrier():
            if torch.distributed.get_rank() == rank:
                world_size = torch.distributed.get_world_size()
                print(f"Rank: [{rank}/{world_size}] Device ID: {device_id}")

    # ================================================================
    # multiprocessing

    multiprocessing.set_start_method(config.multiprocessing.start_method, force=True)

    # ================================================================
    # reproducibility

    config.random.local_seed = (
        config.random.global_seed + torch.distributed.get_rank()
        if config.random.use_unique_seed else config.random.global_seed
    )

    random.seed(config.random.local_seed)
    np.random.seed(config.random.local_seed)
    torch.manual_seed(config.random.local_seed)

    torch.backends.cudnn.benchmark = config.cudnn.benchmark
    torch.backends.cudnn.deterministic = config.cudnn.deterministic

    # ================================================================
    # datasets

    datasets = vsrd.utils.import_module(config.datasets, globals(), locals())

    # ================================================================
    # data loaders

    loaders = vsrd.utils.import_module(config.loaders, globals(), locals())

    # ================================================================
    # utilities

    meters = vsrd.utils.Dict({
        name: vsrd.utils.ProgressMeter(len(loader) * config.optimization.num_steps)
        for name, loader in loaders.items()
    })

    stop_watch = vsrd.utils.StopWatch()

    # ================================================================
    # trainer

    def train():

        stop_watch.start()

        for multi_inputs in vsrd.distributed.tqdm(loaders.train):

            # just for convenience
            multi_inputs = {
                relative_index: vsrd.utils.Dict.apply({
                    key if re.fullmatch(r".*_\dd", key) else inflection.pluralize(key): value
                    for key, value in inputs.items()
                })
                for relative_index, inputs in multi_inputs.items()
            }

            multi_inputs = vsrd.utils.to(multi_inputs, device=device_id, non_blocking=True)

            target_inputs = multi_inputs[0]

            # ================================================================
            # logging

            image_filename, = target_inputs.filenames
            root_dirname = datasets.train.get_root_dirname(image_filename)
            image_dirname = os.path.splitext(os.path.relpath(image_filename, root_dirname))[0]

            logger = vsrd.utils.get_logger(image_dirname)

            config.logging.ckpt_dirname = os.path.join(os.path.dirname(config.config).replace("configs", "ckpts"), image_dirname)
            config.logging.log_dirname = os.path.join(os.path.dirname(config.config).replace("configs", "logs"), image_dirname)
            config.logging.out_dirname = os.path.join(os.path.dirname(config.config).replace("configs", "outs"), image_dirname)

            if os.path.exists(os.path.join(config.logging.ckpt_dirname, f"step_{config.optimization.num_steps - 1}.pt")):
                logger.warning(f"[{image_filename}] Already optimized. Skip this sample.")
                continue

            os.makedirs(config.logging.log_dirname, exist_ok=True)
            log_filename = os.path.join(config.logging.log_dirname, "log.txt")
            file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(levelname)s: %(asctime)s: %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # NOTE: store the main script and config for reproducibility
            shutil.copy(__file__, os.path.join(config.logging.log_dirname, os.path.basename(__file__)))
            with open(os.path.join(config.logging.log_dirname, os.path.basename(config.config)), "w") as file:
                json.dump(config, file, indent=4, sort_keys=False)

            # ================================================================
            # check the number of instances

            num_instances, = map(len, target_inputs.hard_masks)
            if not num_instances:
                logger.warning(f"[{image_filename}] No instances. Skip this sample.")
                continue

            # ================================================================
            # configuration

            logger.info(f"Config: {json.dumps(config, indent=2)}")

            # ================================================================
            # datasets

            dataset_sizes = dict(zip(datasets.keys(), map(len, datasets.values())))

            logger.info(f"Datasets: {json.dumps(dataset_sizes, indent=2)}")

            # ================================================================
            # models

            models = vsrd.utils.import_module(config.models, globals(), locals())

            for model in models.values():
                model.to(device_id)

            logger.info(f"Models: {models}")

            # ================================================================
            # optimizer

            optimizer = vsrd.utils.import_module(config.optimizer, globals(), locals())

            # ================================================================
            # LR scheduler

            scheduler = vsrd.utils.import_module(config.scheduler, globals(), locals())

            # ================================================================
            # summary writer

            writer = torch.utils.tensorboard.SummaryWriter(config.logging.log_dirname)

            # ================================================================
            # checkpoint saver

            saver = vsrd.utils.Saver(config.logging.ckpt_dirname)

            # ================================================================
            # data preparation

            for source_inputs in multi_inputs.values():

                source_instance_indices = [
                    source_instance_ids.new_tensor([
                        source_instance_ids.tolist().index(target_instance_id.item())
                        if target_instance_id in source_instance_ids else -1
                        for target_instance_id in target_instance_ids
                    ])
                    for source_instance_ids, target_instance_ids
                    in zip(source_inputs.instance_ids, target_inputs.instance_ids)
                ]

                source_labels = [
                    vsrd.utils.reversed_pad(source_labels, (0, 1))[source_instance_indices, ...]
                    for source_labels, source_instance_indices
                    in zip(source_inputs.labels, source_instance_indices)
                ]

                source_boxes_2d = [
                    vsrd.utils.reversed_pad(source_boxes_2d, (0, 1))[source_instance_indices, ...]
                    for source_boxes_2d, source_instance_indices
                    in zip(source_inputs.boxes_2d, source_instance_indices)
                ]

                source_boxes_3d = [
                    vsrd.utils.reversed_pad(source_boxes_3d, (0, 1))[source_instance_indices, ...]
                    for source_boxes_3d, source_instance_indices
                    in zip(source_inputs.boxes_3d, source_instance_indices)
                ]

                source_hard_masks = [
                    vsrd.utils.reversed_pad(source_masks, (0, 1))[source_instance_indices, ...]
                    for source_masks, source_instance_indices
                    in zip(source_inputs.hard_masks, source_instance_indices)
                ]

                source_soft_masks = [
                    vsrd.utils.reversed_pad(source_soft_masks, (0, 1))[source_instance_indices, ...]
                    for source_soft_masks, source_instance_indices
                    in zip(source_inputs.soft_masks, source_instance_indices)
                ]

                source_instance_ids = [
                    vsrd.utils.reversed_pad(source_instance_ids, (0, 1))[source_instance_indices, ...]
                    for source_instance_ids, source_instance_indices
                    in zip(source_inputs.instance_ids, source_instance_indices)
                ]

                source_visible_masks = [
                    source_instance_indices.cpu() >= 0
                    for source_instance_indices in source_instance_indices
                ]

                source_inputs.update(
                    labels=source_labels,
                    boxes_2d=source_boxes_2d,
                    boxes_3d=source_boxes_3d,
                    hard_masks=source_hard_masks,
                    soft_masks=source_soft_masks,
                    instance_ids=source_instance_ids,
                    visible_masks=source_visible_masks,
                )

            for inputs in multi_inputs.values():

                camera_positions, ray_directions = vsrd.rendering.ray_casting(
                    image_size=inputs.images.shape[-2:],
                    intrinsic_matrices=inputs.intrinsic_matrices,
                    extrinsic_matrices=inputs.extrinsic_matrices,
                )

                inputs.update(
                    ray_directions=ray_directions,
                    camera_positions=camera_positions,
                )

            multi_ray_directions = list(map(torch.stack, zip(*[
                inputs.ray_directions
                for inputs in multi_inputs.values()
            ])))

            multi_camera_positions = list(map(torch.stack, zip(*[
                torch.stack(list(map(
                    torch.Tensor.expand_as,
                    inputs.camera_positions,
                    inputs.ray_directions,
                )), dim=0)
                for inputs in multi_inputs.values()
            ])))

            multi_hard_masks = list(map(torch.stack, zip(*[
                list(map(
                    functools.partial(torch.permute, dims=(1, 2, 0)),
                    inputs.hard_masks,
                ))
                for inputs in multi_inputs.values()
            ])))

            multi_soft_masks = list(map(torch.stack, zip(*[
                list(map(
                    functools.partial(torch.permute, dims=(1, 2, 0)),
                    inputs.soft_masks,
                ))
                for inputs in multi_inputs.values()
            ])))

            multi_images = list(map(torch.stack, zip(*[
                list(map(
                    functools.partial(torch.permute, dims=(1, 2, 0)),
                    inputs.images,
                ))
                for inputs in multi_inputs.values()
            ])))

            # ================================================================
            # training

            with vsrd.utils.TrainSwitcher(*models.values()):

                for step in vsrd.distributed.tqdm(range(config.optimization.num_steps), leave=False):

                    # ----------------------------------------------------------------
                    # backprop

                    with torch.enable_grad():

                        optimizer.zero_grad()

                        world_outputs = vsrd.utils.Dict.apply(models.detector())

                        # ----------------------------------------------------------------
                        # multi-view projection

                        multi_outputs = vsrd.utils.DefaultDict(vsrd.utils.Dict)

                        world_boxes_3d = nn.functional.pad(world_outputs.boxes_3d, (0, 1), mode="constant", value=1.0)

                        for relative_index, inputs in multi_inputs.items():

                            camera_boxes_3d = torch.einsum("bmn,b...n->b...m", inputs.extrinsic_matrices, world_boxes_3d)
                            camera_boxes_3d = camera_boxes_3d[..., :-1] / camera_boxes_3d[..., -1:]

                            camera_boxes_2d = torch.stack([
                                torch.stack([
                                    vsrd.operations.project_box_3d(
                                        box_3d=camera_box_3d,
                                        line_indices=LINE_INDICES,
                                        intrinsic_matrix=intrinsic_matrix,
                                    )
                                    for camera_box_3d in camera_boxes_3d
                                ], dim=0)
                                for camera_boxes_3d, intrinsic_matrix
                                in zip(camera_boxes_3d, inputs.intrinsic_matrices)
                            ], dim=0)

                            camera_boxes_2d = torchvision.ops.clip_boxes_to_image(
                                boxes=camera_boxes_2d.flatten(-2, -1),
                                size=inputs.images.shape[-2:],
                            ).unflatten(-1, (2, 2))

                            multi_outputs[relative_index].update(
                                boxes_3d=camera_boxes_3d,
                                boxes_2d=camera_boxes_2d,
                            )

                        target_outputs = multi_outputs[0]

                        # ----------------------------------------------------------------
                        # bipartite_matching

                        matching_cost_matrices = [
                            -torchvision.ops.distance_box_iou(
                                boxes1=pd_boxes_2d.flatten(-2, -1),
                                boxes2=gt_boxes_2d.flatten(-2, -1),
                            )
                            for pd_boxes_2d, gt_boxes_2d
                            in zip(target_outputs.boxes_2d, target_inputs.boxes_2d)
                        ]

                        matched_indices = list(map(
                            vsrd.utils.torch_function(sp.optimize.linear_sum_assignment),
                            matching_cost_matrices,
                        ))

                        # ----------------------------------------------------------------
                        # projection loss

                        iou_projection_loss = torch.mean(torch.cat([
                            torch.cat([
                                torchvision.ops.distance_box_iou_loss(
                                    boxes1=pd_boxes_2d[pd_indices[visible_masks[gt_indices]], ...].flatten(-2, -1),
                                    boxes2=gt_boxes_2d[gt_indices[visible_masks[gt_indices]], ...].flatten(-2, -1),
                                    reduction="none",
                                )
                                for pd_boxes_2d, gt_boxes_2d, visible_masks, (pd_indices, gt_indices)
                                in zip(outputs.boxes_2d, inputs.boxes_2d, inputs.visible_masks, matched_indices)
                            ], dim=0)
                            for outputs, inputs in zip(multi_outputs.values(), multi_inputs.values())
                        ], dim=0))

                        l1_projection_loss = torch.mean(torch.cat([
                            torch.cat([
                                nn.functional.smooth_l1_loss(
                                    input=pd_boxes_2d[pd_indices[visible_masks[gt_indices]], ...].flatten(-2, -1),
                                    target=gt_boxes_2d[gt_indices[visible_masks[gt_indices]], ...].flatten(-2, -1),
                                    reduction="none",
                                )
                                for pd_boxes_2d, gt_boxes_2d, visible_masks, (pd_indices, gt_indices)
                                in zip(outputs.boxes_2d, inputs.boxes_2d, inputs.visible_masks, matched_indices)
                            ], dim=0)
                            for outputs, inputs in zip(multi_outputs.values(), multi_inputs.values())
                        ], dim=0))

                        # ----------------------------------------------------------------
                        # instance loss

                        cosine_annealing = lambda x, a, b: (np.cos(np.pi * x) + 1.0) / 2.0 * (a - b) + b
                        cosine_ratio = step / config.optimization.num_steps
                        sdf_union_temperature = cosine_annealing(
                            step / config.optimization.num_steps,
                            config.volume_rendering.max_sdf_union_temperature,
                            config.volume_rendering.min_sdf_union_temperature,
                        )
                        sdf_std_deviation = cosine_annealing(
                            step / config.optimization.num_steps,
                            config.volume_rendering.max_sdf_std_deviation,
                            config.volume_rendering.min_sdf_std_deviation,
                        )

                        def residual_distance_field(distance_field):

                            def wrapper(positions):

                                x_positions, y_positions, z_positions = torch.unbind(positions, dim=-1)
                                positions = torch.stack([torch.abs(x_positions), y_positions, z_positions], dim=-1)

                                # positions = torch.tanh(positions / torch.max(models.detector.dimension_range, dim=0).values)
                                positions = positions / max(config.volume_rendering.distance_range)
                                positions = models.positional_encoder(positions)

                                distances = distance_field(positions)
                                distances = torch.sigmoid(distances - 1.0)

                                return distances

                            return wrapper

                        def residual_composition(distance_field, residual_distance_field):

                            def wrapper(positions):
                                distances = distance_field(positions)
                                residuals = residual_distance_field(positions)
                                return distances + residuals

                            return wrapper

                        def instance_field(distance_field, instance_label):

                            def wrapper(positions):

                                distances = distance_field(positions)

                                # positions = torch.tanh(positions / torch.max(models.detector.dimension_range, dim=0).values)
                                positions = positions / max(config.volume_rendering.distance_range)
                                positions = models.positional_encoder(positions)

                                instance_labels = nn.functional.one_hot(instance_label, num_instances)
                                instance_labels = instance_labels.expand(*distances.shape[:-1], -1)

                                return distances, instance_labels

                            return wrapper

                        def soft_union(distance_fields, temperature):

                            def wrapper(positions):
                                distances, *multi_features = map(torch.stack, zip(*[
                                    distance_field(positions)
                                    for distance_field in distance_fields
                                ]))
                                weights = nn.functional.softmin(distances / temperature, dim=0)
                                distances = torch.sum(distances * weights, dim=0)
                                multi_features = [
                                    torch.sum(features * weights, dim=0)
                                    for features in multi_features
                                ]
                                return distances, *multi_features

                            return wrapper

                        def hard_union(distance_fields):

                            def wrapper(positions):
                                distances, *multi_features = map(torch.stack, zip(*[
                                    distance_field(positions)
                                    for distance_field in distance_fields
                                ]))
                                indices = torch.argmin(distances, dim=0, keepdim=True)
                                distances = torch.gather(distances, index=indices, dim=0).squeeze(0)
                                multi_features = [
                                    torch.gather(features, index=indices.expand(*indices.shape[:-1], *features.shape[-1:]), dim=0).squeeze(0)
                                    for features in multi_features
                                ]
                                return distances, *multi_features

                            return wrapper

                        def hierarchical_wrapper(renderer):

                            def wrapper(*args, **kwargs):

                                with torch.no_grad():
                                    *_, sampled_distances, sampled_weights = renderer(*args, **kwargs)

                                kwargs.update(sampled_distances=sampled_distances, sampled_weights=sampled_weights)
                                *outputs, _, _ = renderer(*args, **kwargs)

                                return outputs

                            return wrapper

                        if step >= config.optimization.warmup_steps:

                            distance_field_weights = models.hyper_distance_field(world_outputs.embeddings)
                            world_outputs.update(distance_field_weights=distance_field_weights)

                            soft_distance_fields = [
                                soft_union(
                                    distance_fields=[
                                        vsrd.rendering.sdfs.translation(
                                            vsrd.rendering.sdfs.rotation(
                                                instance_field(
                                                    distance_field=residual_composition(
                                                        distance_field=vsrd.rendering.sdfs.box(dimension),
                                                        residual_distance_field=residual_distance_field(
                                                            distance_field=functools.partial(
                                                                models.hyper_distance_field.distance_field,
                                                                distance_field_weights,
                                                            ),
                                                        ),
                                                    ),
                                                    instance_label=dimension.new_tensor(instance_label, dtype=torch.long),
                                                ),
                                                orientation,
                                            ),
                                            location,
                                        )
                                        for instance_label, (
                                            location,
                                            dimension,
                                            orientation,
                                            distance_field_weights,
                                        )
                                        in enumerate(zip(
                                            locations,
                                            dimensions,
                                            orientations,
                                            distance_field_weights,
                                        ))
                                    ],
                                    temperature=sdf_union_temperature,
                                )
                                for (
                                    locations,
                                    dimensions,
                                    orientations,
                                    distance_field_weights,
                                )
                                in zip(
                                    world_outputs.locations,
                                    world_outputs.dimensions,
                                    world_outputs.orientations,
                                    world_outputs.distance_field_weights,
                                )
                            ]

                        else:

                            soft_distance_fields = [
                                soft_union(
                                    distance_fields=[
                                        vsrd.rendering.sdfs.translation(
                                            vsrd.rendering.sdfs.rotation(
                                                instance_field(
                                                    distance_field=vsrd.rendering.sdfs.box(dimension),
                                                    instance_label=dimension.new_tensor(instance_label, dtype=torch.long),
                                                ),
                                                orientation,
                                            ),
                                            location,
                                        )
                                        for instance_label, (
                                            location,
                                            dimension,
                                            orientation,
                                        )
                                        in enumerate(zip(
                                            locations,
                                            dimensions,
                                            orientations,
                                        ))
                                    ],
                                    temperature=sdf_union_temperature,
                                )
                                for (
                                    locations,
                                    dimensions,
                                    orientations,
                                )
                                in zip(
                                    world_outputs.locations,
                                    world_outputs.dimensions,
                                    world_outputs.orientations,
                                )
                            ]

                        multi_ray_indices = torch.multinomial(
                            input=torch.stack([
                                torch.max(multi_soft_masks, dim=-1).values
                                for multi_soft_masks in multi_soft_masks
                            ], dim=0).flatten(1, -1),
                            num_samples=config.volume_rendering.num_rays,
                            replacement=False,
                        )

                        multi_sampled_labels, multi_sampled_gradients = map(torch.stack, zip(*[
                            hierarchical_wrapper(vsrd.rendering.hierarchical_volumetric_rendering)(
                                distance_field=soft_distance_field,
                                ray_positions=multi_camera_positions.flatten(0, -2)[multi_ray_indices, ...],
                                ray_directions=multi_ray_directions.flatten(0, -2)[multi_ray_indices, ...],
                                distance_range=config.volume_rendering.distance_range,
                                num_samples=config.volume_rendering.num_fine_samples,
                                sdf_std_deviation=sdf_std_deviation,
                                cosine_ratio=cosine_ratio,
                            )
                            for (
                                soft_distance_field,
                                multi_camera_positions,
                                multi_ray_directions,
                                multi_ray_indices,
                            )
                            in zip(
                                soft_distance_fields,
                                multi_camera_positions,
                                multi_ray_directions,
                                multi_ray_indices,
                            )
                        ]))

                        silhouette_loss = torch.mean(torch.stack([
                            nn.functional.binary_cross_entropy(
                                input=multi_sampled_labels[..., pd_indices].clamp(1.0e-6, 1.0 - 1.0e-6),
                                target=multi_soft_masks.flatten(0, -2)[multi_ray_indices, ...][..., gt_indices],
                                reduction="none",
                            )
                            for (
                                multi_sampled_labels,
                                multi_soft_masks,
                                multi_ray_indices,
                                (pd_indices, gt_indices),
                            )
                            in zip(
                                multi_sampled_labels,
                                multi_soft_masks,
                                multi_ray_indices,
                                matched_indices,
                            )
                        ], dim=0))

                        losses = vsrd.utils.Dict(
                            iou_projection_loss=iou_projection_loss,
                            l1_projection_loss=l1_projection_loss,
                            silhouette_loss=silhouette_loss,
                        )

                        if step >= config.optimization.warmup_steps:

                            eikonal_loss = nn.functional.mse_loss(
                                input=torch.norm(multi_sampled_gradients, dim=-1),
                                target=multi_sampled_gradients.new_ones(*multi_sampled_gradients.shape[:-1]),
                                reduction="mean",
                            )

                            losses.update(eikonal_loss=eikonal_loss)

                            if config.loss_weights.photometric_loss:

                                hard_distance_fields = [
                                    hard_union([
                                        vsrd.rendering.sdfs.translation(
                                            vsrd.rendering.sdfs.rotation(
                                                instance_field(
                                                    distance_field=residual_composition(
                                                        distance_field=vsrd.rendering.sdfs.box(dimension),
                                                        residual_distance_field=residual_distance_field(
                                                            distance_field=functools.partial(
                                                                models.hyper_distance_field.distance_field,
                                                                distance_field_weights,
                                                            ),
                                                        ),
                                                    ),
                                                    instance_label=dimension.new_tensor(instance_label, dtype=torch.long),
                                                ),
                                                orientation,
                                            ),
                                            location,
                                        )
                                        for instance_label, (
                                            location,
                                            dimension,
                                            orientation,
                                            distance_field_weights,
                                        )
                                        in enumerate(zip(
                                            locations,
                                            dimensions,
                                            orientations,
                                            distance_field_weights,
                                        ))
                                    ])
                                    for (
                                        locations,
                                        dimensions,
                                        orientations,
                                        distance_field_weights,
                                    )
                                    in zip(
                                        world_outputs.locations,
                                        world_outputs.dimensions,
                                        world_outputs.orientations,
                                        world_outputs.distance_field_weights,
                                    )
                                ]

                                target_ray_indices = torch.multinomial(
                                    input=torch.stack([
                                        torch.max(hard_masks, dim=0).values
                                        for hard_masks in target_inputs.hard_masks
                                    ], dim=0).flatten(1, -1),
                                    num_samples=config.surface_rendering.num_rays,
                                    replacement=False,
                                )

                                target_sampled_positions, target_convergence_masks = map(torch.stack, zip(*[
                                    vsrd.rendering.sphere_tracing(
                                        distance_field=vsrd.utils.compose(hard_distance_field, operator.itemgetter(0)),
                                        ray_positions=camera_position.expand(*ray_indices.shape, -1),
                                        ray_directions=ray_directions.flatten(0, -2)[ray_indices, ...],
                                        num_iterations=config.surface_rendering.num_iterations,
                                        convergence_criteria=config.surface_rendering.convergence_criteria,
                                        bounding_radius=config.surface_rendering.bounding_radius,
                                        initialization=False,
                                        differentiable=True,
                                    )
                                    for (
                                        hard_distance_field,
                                        camera_position,
                                        ray_directions,
                                        ray_indices,
                                    )
                                    in zip(
                                        hard_distance_fields,
                                        target_inputs.camera_positions,
                                        target_inputs.ray_directions,
                                        target_ray_indices,
                                    )
                                ]))

                                if torch.any(target_convergence_masks):

                                    target_sampled_normals = torch.stack([
                                        vsrd.rendering.surface_normal(
                                            distance_field=vsrd.utils.compose(hard_distance_field, operator.itemgetter(0)),
                                            surface_positions=target_sampled_positions,
                                        )
                                        for hard_distance_field, target_sampled_positions
                                        in zip(hard_distance_fields, target_sampled_positions)
                                    ], dim=0)

                                    relative_grids = torch.stack(torch.meshgrid(*[
                                        torch.arange(patch_size, dtype=torch.float, device=device_id) - patch_size // 2
                                        for patch_size in config.surface_rendering.patch_size
                                    ], indexing="xy"), dim=-1)

                                    target_coordinates = torch.stack([
                                        target_ray_indices % target_inputs.images.shape[-1],
                                        target_ray_indices // target_inputs.images.shape[-1],
                                    ], dim=-1).unsqueeze(-2).unsqueeze(-3) + relative_grids

                                    target_coordinates = nn.functional.pad(target_coordinates, (0, 1), mode="constant", value=1.0)

                                    for relative_index, source_inputs in multi_inputs.items():

                                        source_homographies = torch.stack([
                                            Ks @ (Rs @ Rt.T - Rs @ (Rs.T @ ts - Rt.T @ tt) @ nT / -(nT @ p)) @ torch.linalg.inv(Kt)
                                            for Ks, Rs, ts, Kt, Rt, tt, nT, p
                                            in zip(
                                                source_inputs.intrinsic_matrices,
                                                source_inputs.extrinsic_matrices[..., :3, :3],
                                                source_inputs.extrinsic_matrices[..., :3, 3:],
                                                target_inputs.intrinsic_matrices,
                                                target_inputs.extrinsic_matrices[..., :3, :3],
                                                target_inputs.extrinsic_matrices[..., :3, 3:],
                                                target_sampled_normals.unsqueeze(-1).transpose(-2, -1),
                                                target_sampled_positions.unsqueeze(-1),
                                            )
                                        ], dim=0)

                                        source_coordinates = torch.einsum(f"blmn,bl...n->bl...m", source_homographies, target_coordinates)
                                        source_coordinates = source_coordinates[..., :-1] / (source_coordinates[..., -1:] + 1e-6)
                                        source_coordinates = vsrd.utils.linear_map(source_coordinates, 0, np.subtract(source_inputs.images.shape[-2:][::-1], 1), -1.0, 1.0)

                                        sampled_patches = nn.functional.grid_sample(
                                            input=source_inputs.images,
                                            grid=source_coordinates.flatten(-3, -2),
                                            mode="bilinear",
                                            padding_mode="zeros",
                                        )

                                        sampled_patches = torchvision.transforms.functional.rgb_to_grayscale(sampled_patches)

                                        multi_outputs[relative_index].update(sampled_patches=sampled_patches)

                                    def corrcoef(inputs, *args, epsilon=1e-12, **kwargs):
                                        covariances = torch.cov(inputs, *args, **kwargs)
                                        variances = torch.diag(covariances)
                                        variances = variances.unsqueeze(-1) * variances.unsqueeze(0)
                                        variances = torch.sqrt(variances + epsilon)
                                        return covariances / variances

                                    sampled_patches = list(map(torch.stack, zip(*[
                                        [
                                            source_sampled_patches[..., target_convergence_masks, :]
                                            for source_sampled_patches, target_convergence_masks
                                            in zip(source_outputs.sampled_patches, target_convergence_masks.squeeze(-1))
                                        ]
                                        for source_outputs in multi_outputs.values()
                                    ])))

                                    coefficient_matrices = torch.cat(sum([
                                        [
                                            torch.topk(corrcoef(sampled_patches.flatten(1, -1)), k=4, dim=-1).values
                                            for sampled_patches in torch.unbind(sampled_patches, dim=-2)
                                        ]
                                        for sampled_patches in sampled_patches
                                    ], []), dim=0)

                                    photometric_loss = (1.0 - torch.nanmean(coefficient_matrices)) / 2.0

                                    losses.update(photometric_loss=photometric_loss)

                        loss = sum(loss * config.loss_weights[name] for name, loss in losses.items())

                        meters.train.update(forward=stop_watch.restart())

                        torch.autograd.backward(loss)

                        meters.train.update(backward=stop_watch.restart())

                        optimizer.step()

                        scheduler.step()

                    # ----------------------------------------------------------------
                    # logging

                    with torch.no_grad():

                        if not (step + 1) % config.logging.scalar_intervals:

                            # ----------------------------------------------------------------
                            # evaluation

                            pd_boxes_3d = [
                                pd_boxes_3d[pd_indices, ...] @ rectification_matrix.T
                                for pd_boxes_3d, rectification_matrix, (pd_indices, _)
                                in zip(target_outputs.boxes_3d, target_inputs.rectification_matrices, matched_indices)
                            ]
                            gt_boxes_3d = [
                                gt_boxes_3d[gt_indices, ...] @ rectification_matrix.T
                                for gt_boxes_3d, rectification_matrix, (_, gt_indices)
                                in zip(target_inputs.boxes_3d, target_inputs.rectification_matrices, matched_indices)
                            ]

                            metrics = {}

                            if any([any(map(vsrd.utils.compose(torch.isfinite, torch.all), gt_boxes_3d)) for gt_boxes_3d in gt_boxes_3d]):

                                rotation_matrix = vsrd.operations.rotation_matrix_x(torch.tensor(-np.pi / 2.0, device=device_id))

                                ious_3d, ious_bev = map(torch.as_tensor, zip(*sum([
                                    [
                                        vsrd.operations.box_3d_iou(
                                            corners1=pd_box_3d @ rotation_matrix.T,
                                            corners2=gt_box_3d @ rotation_matrix.T,
                                        )
                                        for pd_box_3d, gt_box_3d
                                        in zip(pd_boxes_3d, gt_boxes_3d)
                                        if torch.all(torch.isfinite(gt_box_3d))
                                    ]
                                    for pd_boxes_3d, gt_boxes_3d
                                    in zip(pd_boxes_3d, gt_boxes_3d)
                                ], [])))

                                iou_3d = torch.mean(ious_3d)
                                iou_bev = torch.mean(ious_bev)

                                accuracy_3d_25 = torch.mean((ious_3d > 0.25).float())
                                accuracy_bev_25 = torch.mean((ious_bev > 0.25).float())

                                accuracy_3d_50 = torch.mean((ious_3d > 0.50).float())
                                accuracy_bev_50 = torch.mean((ious_bev > 0.50).float())

                                metrics.update(
                                    iou_3d=iou_3d,
                                    iou_bev=iou_bev,
                                    accuracy_3d_25=accuracy_3d_25,
                                    accuracy_bev_25=accuracy_bev_25,
                                    accuracy_3d_50=accuracy_3d_50,
                                    accuracy_bev_50=accuracy_bev_50,
                                )

                            scalars = {
                                **{f"losses/{name}": loss.item() for name, loss in losses.items()},
                                **{f"metrics/{name}": metric.item() for name, metric in metrics.items()},
                                **{
                                    f"learning_rates/{param_group_name}": param_group["lr"]
                                    for param_group_name, param_group
                                    in zip(config.optimization.param_group_names, optimizer.param_groups)
                                },
                                **{
                                    f"hyperparameters/sdf_union_temperature": sdf_union_temperature,
                                    f"hyperparameters/sdf_std_deviation": sdf_std_deviation,
                                    f"hyperparameters/cosine_ratio": cosine_ratio,
                                },
                            }

                            logger.info(
                                f"[Training] Rank: {torch.distributed.get_rank()}, Step: {step}, Progress: {meters.train.progress():.2%}, "
                                f"ETA: {datetime.timedelta(seconds=meters.train.arrival_seconds())}, "
                                f"scalars: {json.dumps(scalars, indent=4)}"
                            )
                            logger.info(
                                f"[Training] Rank: {torch.distributed.get_rank()}, Step: {step}, Progress: {meters.train.progress():.2%}, "
                                f"ETA: {datetime.timedelta(seconds=meters.train.arrival_seconds())}, "
                                f"runtimes: {json.dumps(dict(zip(meters.train.keys(), meters.train.means())), indent=4)}"
                            )

                            for name, metric in scalars.items():
                                writer.add_scalar(f"scalars/{name}", metric, step)

                        if not (step + 1) % config.logging.image_intervals:

                            for frame_index, ((relative_index, inputs), (relative_index, outputs)) \
                                in enumerate(zip(multi_inputs.items(), multi_outputs.items())):

                                images = {}

                                def get_pixel_indices(frame_indices, *pixel_indices):
                                    return tuple(map(operator.itemgetter(frame_indices == frame_index), pixel_indices))

                                pixel_indices = [
                                    get_pixel_indices(*vsrd.utils.torch_function(np.unravel_index)(
                                        indices=multi_ray_indices,
                                        shape=multi_soft_masks.shape[:-1],
                                    ))
                                    for multi_ray_indices, multi_soft_masks
                                    in zip(multi_ray_indices, multi_soft_masks)
                                ]

                                gt_images = torch.stack([
                                    vsrd.visualization.draw_points_2d(
                                        image=vsrd.visualization.draw_boxes_3d(
                                            image=vsrd.visualization.draw_masks(image, gt_masks),
                                            boxes_3d=gt_boxes_3d[torch.all(torch.isfinite(gt_boxes_3d.flatten(-2, -1)), dim=-1), ...],
                                            line_indices=LINE_INDICES + [[0, 5], [1, 4]],
                                            intrinsic_matrix=intrinsic_matrix,
                                            color=(255, 255, 255),
                                            thickness=2,
                                            lineType=cv.LINE_AA,
                                        ),
                                        points_2d=torch.stack(list(reversed(pixel_indices)), dim=-1),
                                        color=(255, 0, 0),
                                        radius=2,
                                        thickness=-1,
                                        lineType=cv.LINE_AA,
                                    )
                                    for (
                                        image,
                                        gt_masks,
                                        gt_boxes_3d,
                                        intrinsic_matrix,
                                        pixel_indices,
                                    )
                                    in zip(
                                        inputs.images,
                                        inputs.soft_masks,
                                        inputs.boxes_3d,
                                        inputs.intrinsic_matrices,
                                        pixel_indices,
                                    )
                                ], dim=0)

                                images.update(gt_images=gt_images)

                                if not relative_index:

                                    volume_masks = torch.stack([
                                        torch.stack([
                                            hierarchical_wrapper(vsrd.rendering.hierarchical_volumetric_rendering)(
                                                distance_field=soft_distance_field,
                                                ray_positions=camera_position,
                                                ray_directions=ray_directions,
                                                distance_range=config.volume_rendering.distance_range,
                                                num_samples=config.volume_rendering.num_fine_samples,
                                                sdf_std_deviation=sdf_std_deviation,
                                                cosine_ratio=cosine_ratio,
                                            )[0]
                                            for ray_directions in ray_directions
                                        ], dim=0).permute(2, 0, 1)
                                        for soft_distance_field, camera_position, ray_directions
                                        in zip(soft_distance_fields, inputs.camera_positions, inputs.ray_directions)
                                    ], dim=0)

                                    surface_masks = torch.stack([
                                        vsrd.rendering.sphere_tracing(
                                            distance_field=vsrd.utils.compose(soft_distance_field, operator.itemgetter(0)),
                                            ray_positions=camera_position,
                                            ray_directions=ray_directions,
                                            num_iterations=config.surface_rendering.num_iterations,
                                            convergence_criteria=config.surface_rendering.convergence_criteria,
                                            bounding_radius=config.surface_rendering.bounding_radius,
                                            initialization=False,
                                            differentiable=False,
                                        )[1].permute(2, 0, 1)
                                        for soft_distance_field, camera_position, ray_directions
                                        in zip(soft_distance_fields, inputs.camera_positions, inputs.ray_directions)
                                    ], dim=0)

                                    pd_images = torch.stack([
                                        vsrd.visualization.draw_boxes_3d(
                                            image=vsrd.visualization.draw_masks(image, pd_masks),
                                            boxes_3d=pd_boxes_3d,
                                            line_indices=LINE_INDICES + [[0, 5], [1, 4]],
                                            intrinsic_matrix=intrinsic_matrix,
                                            color=(255, 255, 255),
                                            thickness=2,
                                            lineType=cv.LINE_AA,
                                        )
                                        for (
                                            image,
                                            pd_masks,
                                            pd_boxes_3d,
                                            intrinsic_matrix,
                                        )
                                        in zip(
                                            inputs.images,
                                            volume_masks * surface_masks,
                                            outputs.boxes_3d,
                                            inputs.intrinsic_matrices,
                                        )
                                    ], dim=0)

                                    bev_images = torch.stack([
                                        vsrd.visualization.draw_boxes_bev(
                                            image=vsrd.visualization.draw_boxes_bev(
                                                image=image.new_ones(3, 1000, 1000),
                                                boxes_3d=(
                                                    gt_boxes_3d[torch.all(torch.isfinite(gt_boxes_3d.flatten(-2, -1)), dim=-1), ...] @
                                                    rectification_matrix.T
                                                ),
                                                color=(255, 0, 0),
                                                thickness=2,
                                                lineType=cv.LINE_AA,
                                            ),
                                            boxes_3d=(
                                                pd_boxes_3d @
                                                rectification_matrix.T
                                            ),
                                            color=(0, 0, 255),
                                            thickness=2,
                                            lineType=cv.LINE_AA,
                                        )
                                        for (
                                            image,
                                            gt_boxes_3d,
                                            pd_boxes_3d,
                                            rectification_matrix,
                                        )
                                        in zip(
                                            inputs.images,
                                            inputs.boxes_3d,
                                            outputs.boxes_3d,
                                            inputs.rectification_matrices,
                                        )
                                    ], dim=0)

                                    images.update(
                                        pd_images=pd_images,
                                        bev_images=bev_images,
                                    )

                                for name, image in images.items():
                                    writer.add_images(f"images/{name}/{relative_index:+}", image, step)

                        if not (step + 1) % config.logging.ckpt_intervals:

                            saver.save(
                                filename=f"step_{step}.pt",
                                step=step,
                                models={
                                    name: model.state_dict()
                                    for name, model in models.items()
                                },
                                optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),
                                metrics=metrics,
                            )

                        meters.train.update(logging=stop_watch.restart())

        stop_watch.stop()


    # ================================================================
    # main

    config.train and train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
        "VSRD: Volumetric Silhouette Rendering"
        " for Monocular 3D Object Detection without 3D Supervision"
    )
    parser.add_argument("--launcher", type=str, choices=["slurm", "torch"], default="slurm")
    parser.add_argument("--port", type=int, default=1209)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--config", type=str)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    main(args)
