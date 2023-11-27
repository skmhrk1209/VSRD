import operator

import torch
import torch.nn as nn

from .. import utils
from . import samplers


def sphere_intersection(ray_positions, ray_directions, bounding_radius):
    a = torch.sum(ray_directions * ray_directions, dim=-1, keepdim=True)
    b = torch.sum(ray_directions * ray_positions, dim=-1, keepdim=True)
    c = torch.sum(ray_positions * ray_positions, dim=-1, keepdim=True) - bounding_radius ** 2.0
    d = b ** 2.0 - a * c
    intersection_masks = d >= 0.0
    min_ray_distances = (-b - torch.sqrt(d)) / a
    max_ray_distances = (-b + torch.sqrt(d)) / a
    return min_ray_distances, max_ray_distances, intersection_masks


def sphere_tracing(
    distance_field,
    ray_positions,
    ray_directions,
    num_iterations,
    convergence_criteria,
    foreground_masks=None,
    bounding_radius=None,
    initialization=True,
    differentiable=False,
):
    if foreground_masks is None:
        foreground_masks = torch.all(torch.isfinite(ray_positions), dim=-1, keepdim=True)

    if bounding_radius and initialization:
        min_ray_distances, _, intersection_masks = sphere_intersection(ray_positions, ray_directions, bounding_radius)
        ray_positions = torch.where(
            intersection_masks,
            ray_positions + ray_directions * min_ray_distances,
            ray_positions,
        )
        foreground_masks = foreground_masks & intersection_masks

    with torch.no_grad():
        convergence_masks = torch.zeros_like(foreground_masks)
        for _ in range(num_iterations):
            signed_distances = distance_field(ray_positions)
            ray_positions = torch.where(
                foreground_masks & ~convergence_masks,
                ray_positions + ray_directions * signed_distances,
                ray_positions,
            )
            if bounding_radius:
                intersection_masks = torch.linalg.norm(ray_positions, dim=-1, keepdim=True) < bounding_radius
                foreground_masks = foreground_masks & intersection_masks
            convergence_masks = torch.abs(signed_distances) < convergence_criteria
            if torch.all(~foreground_masks | convergence_masks): break

    if differentiable:
        retain_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            ray_positions.requires_grad_(True)
            signed_distances = distance_field(ray_positions)
            gradients, = torch.autograd.grad(
                outputs=signed_distances,
                inputs=ray_positions,
                grad_outputs=torch.ones_like(signed_distances),
                retain_graph=retain_graph,
            )
            signed_distances = -signed_distances / torch.sum(gradients * ray_directions, dim=-1, keepdim=True)
            ray_positions = torch.where(convergence_masks, ray_positions + ray_directions * signed_distances, ray_positions)

    return ray_positions, convergence_masks


def surface_normal(
    distance_field,
    surface_positions,
    finite_difference_epsilon=None,
):
    if finite_difference_epsilon:
        finite_difference_epsilon_x = surface_positions.new_tensor([finite_difference_epsilon, 0.0, 0.0])
        finite_difference_epsilon_y = surface_positions.new_tensor([0.0, finite_difference_epsilon, 0.0])
        finite_difference_epsilon_z = surface_positions.new_tensor([0.0, 0.0, finite_difference_epsilon])
        surface_normals_x = (
            distance_field(surface_positions + finite_difference_epsilon_x) -
            distance_field(surface_positions - finite_difference_epsilon_x)
        )
        surface_normals_y = (
            distance_field(surface_positions + finite_difference_epsilon_y) -
            distance_field(surface_positions - finite_difference_epsilon_y)
        )
        surface_normals_z = (
            distance_field(surface_positions + finite_difference_epsilon_z) -
            distance_field(surface_positions - finite_difference_epsilon_z)
        )
        surface_normals = torch.cat((surface_normals_x, surface_normals_y, surface_normals_z), dim=-1)

    else:
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            surface_positions.requires_grad_(True)
            signed_distances = distance_field(surface_positions)
            surface_normals, = torch.autograd.grad(
                outputs=signed_distances,
                inputs=surface_positions,
                grad_outputs=torch.ones_like(signed_distances),
                create_graph=create_graph,
            )

    surface_normals = nn.functional.normalize(surface_normals, dim=-1)

    return surface_normals


def phong_shading(
    ray_directions,
    surface_normals,
    light_directions,
    light_ambient_colors,
    light_diffuse_colors,
    light_specular_colors,
    material_ambient_colors,
    material_diffuse_colors,
    material_specular_colors,
    material_emission_colors,
    material_shininesses,
):
    ray_directions = nn.functional.normalize(ray_directions, dim=-1)
    surface_normals = nn.functional.normalize(surface_normals, dim=-1)
    light_directions = nn.functional.normalize(light_directions, dim=-1)

    reflected_directions = light_directions - 2.0 * surface_normals * torch.sum(light_directions * surface_normals, dim=-1, keepdim=True)

    diffuse_coefficients = nn.functional.relu(-torch.sum(light_directions * surface_normals, dim=-1, keepdim=True))
    specular_coefficients = nn.functional.relu(-torch.sum(reflected_directions * ray_directions, dim=-1, keepdim=True)) ** material_shininesses

    emission_colors = material_emission_colors
    ambient_colors = material_ambient_colors * light_ambient_colors
    diffuse_colors = material_diffuse_colors * light_diffuse_colors * diffuse_coefficients
    specular_colors = material_specular_colors * light_specular_colors * specular_coefficients
    colors = emission_colors + ambient_colors + diffuse_colors + specular_colors

    colors = torch.clamp(colors, 0.0, 1.0)

    return colors


def shadow_rendering(
    distance_field,
    surface_positions,
    surface_normals,
    light_directions,
    num_iterations,
    convergence_criteria,
    foreground_masks,
    bounding_radius=None,
    initialization=False,
    implicit_differentiation=False,
):
    ray_positions = surface_positions + surface_normals * convergence_criteria
    ray_directions = -light_directions
    surface_positions, convergence_masks = sphere_tracing(
        distance_field=distance_field,
        ray_positions=ray_positions,
        ray_directions=ray_directions,
        num_iterations=num_iterations,
        convergence_criteria=convergence_criteria,
        foreground_masks=foreground_masks,
        bounding_radius=bounding_radius,
        initialization=initialization,
        differentiable=implicit_differentiation,
    )
    return foreground_masks & convergence_masks


def hierarchical_volumetric_rendering(
    distance_field,
    ray_positions,
    ray_directions,
    distance_range,
    num_samples,
    sdf_std_deviation,
    cosine_ratio=1.0,
    epsilon=1e-6,
    sampled_distances=None,
    sampled_weights=None,
):
    if sampled_distances is None:

        distance_bins = torch.linspace(*distance_range, num_samples + 1, device=ray_directions.device)
        distance_bins = distance_bins.expand(*ray_directions.shape[:-1], 1, -1)

        sampled_distances = samplers.quadrature_sampler(distance_bins)

    else:

        sampled_distances = sampled_distances.permute(*range(1, sampled_distances.ndim), 0)
        sampled_weights = sampled_weights.permute(*range(1, sampled_weights.ndim), 0)

        sampled_distances = torch.cat([
            sampled_distances,
            samplers.inverse_transform_sampler(
                bins=sampled_distances,
                weights=sampled_weights,
                num_samples=num_samples,
            ),
        ], dim=-1)

        sampled_distances = torch.sort(sampled_distances, dim=-1, descending=False).values

    sampled_distances = sampled_distances.permute(-1, *range(sampled_distances.ndim - 1))
    sampled_intervals = sampled_distances[1:, ...] - sampled_distances[:-1, ...]
    sampled_midpoints = (sampled_distances[:-1, ...] + sampled_distances[1:, ...]) / 2.0

    sampled_positions = ray_positions + ray_directions * sampled_midpoints

    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        sampled_positions.requires_grad_(True)
        sampled_signed_distances, *multi_sampled_features = distance_field(sampled_positions)
        sampled_gradients, = torch.autograd.grad(
            outputs=sampled_signed_distances,
            inputs=sampled_positions,
            grad_outputs=torch.ones_like(sampled_signed_distances),
            create_graph=create_graph,
        )
        sampled_normals = nn.functional.normalize(sampled_gradients, dim=-1)

    # NOTE: why...?
    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations.
    # The anneal strategy below makes the cos value "not dead" at the beginning training iterations, for better convergence.
    # https://github.com/bennyguo/instant-nsr-pl/blob/master/models/neus.py#L125
    sampled_cosines = torch.sum(ray_directions * sampled_normals, dim=-1, keepdim=True)
    sampled_cosines = -torch.lerp(
        nn.functional.relu(-sampled_cosines * 0.5 + 0.5),
        nn.functional.relu(-sampled_cosines),
        cosine_ratio,
    )

    # NOTE: estimate signed distances at section points
    # https://github.com/Totoro97/NeuS/blob/master/models/renderer.py#L237
    prev_sampled_signed_distances = sampled_signed_distances - sampled_cosines * sampled_intervals / 2.0
    next_sampled_signed_distances = sampled_signed_distances + sampled_cosines * sampled_intervals / 2.0

    prev_sampled_cdf = torch.sigmoid(prev_sampled_signed_distances / sdf_std_deviation)
    next_sampled_cdf = torch.sigmoid(next_sampled_signed_distances / sdf_std_deviation)
    sampled_opacities = nn.functional.relu((prev_sampled_cdf - next_sampled_cdf) / (prev_sampled_cdf + epsilon))

    accumulated_transmittances = torch.cumprod(1.0 - sampled_opacities, dim=0)
    # NOTE: `exclusive` parameter in `tf.math.cumprod`` is not supported in PyTorch
    # https://github.com/bmild/nerf/blob/master/run_nerf.py#L143
    accumulated_transmittances = torch.cat([
        torch.ones_like(accumulated_transmittances[:1, ...]),
        accumulated_transmittances[:-1, ...],
    ], dim=0)

    sampled_weights = accumulated_transmittances * sampled_opacities

    multi_accumulated_features = [
        torch.sum(sampled_features * sampled_weights, dim=0)
        for sampled_features in multi_sampled_features
    ]

    return (
        *multi_accumulated_features,
        sampled_gradients,
        sampled_distances,
        sampled_weights,
    )


def occupancy_volumetric_rendering(
    distance_field,
    occupancy_grid,
    ray_positions,
    ray_directions,
    marching_step_size,
    sdf_std_deviation,
    cosine_ratio=1.0,
    stratified=True,
    epsilon=1e-6,
):
    import nerfacc

    # NOTE: just for NerfAcc
    def opacity_evaluator(prev_sampled_distances, next_sampled_distances, ray_indices):

        prev_sampled_distances = prev_sampled_distances.unsqueeze(-1)
        next_sampled_distances = next_sampled_distances.unsqueeze(-1)

        sampled_ray_positions = ray_positions[ray_indices, ...]
        sampled_ray_directions = ray_directions[ray_indices, ...]

        sampled_intervals = next_sampled_distances - prev_sampled_distances
        sampled_midpoints = (prev_sampled_distances + next_sampled_distances) / 2.0

        sampled_positions = sampled_ray_positions + sampled_ray_directions * sampled_midpoints

        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            sampled_positions.requires_grad_(True)
            sampled_signed_distances, *multi_sampled_features = distance_field(sampled_positions)
            sampled_gradients, = torch.autograd.grad(
                outputs=sampled_signed_distances,
                inputs=sampled_positions,
                grad_outputs=torch.ones_like(sampled_signed_distances),
                create_graph=create_graph,
            )
            sampled_normals = nn.functional.normalize(sampled_gradients, dim=-1)

        # NOTE: why...?
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations.
        # The anneal strategy below makes the cos value "not dead" at the beginning training iterations, for better convergence.
        # https://github.com/bennyguo/instant-nsr-pl/blob/master/models/neus.py#L125
        sampled_cosines = torch.sum(sampled_ray_directions * sampled_normals, dim=-1, keepdim=True)
        sampled_cosines = -torch.lerp(
            nn.functional.relu(-sampled_cosines * 0.5 + 0.5),
            nn.functional.relu(-sampled_cosines),
            cosine_ratio,
        )

        # NOTE: estimate signed distances at section points
        # https://github.com/Totoro97/NeuS/blob/master/models/renderer.py#L237
        prev_sampled_signed_distances = sampled_signed_distances - sampled_cosines * sampled_intervals / 2.0
        next_sampled_signed_distances = sampled_signed_distances + sampled_cosines * sampled_intervals / 2.0

        prev_sampled_cdf = torch.sigmoid(prev_sampled_signed_distances / sdf_std_deviation)
        next_sampled_cdf = torch.sigmoid(next_sampled_signed_distances / sdf_std_deviation)
        sampled_opacities = nn.functional.relu((prev_sampled_cdf - next_sampled_cdf) / (prev_sampled_cdf + epsilon))

        sampled_opacities = sampled_opacities.squeeze(-1)

        return sampled_opacities, sampled_gradients, *multi_sampled_features

    ray_indices, prev_ray_distances, next_ray_distances = occupancy_grid.sampling(
        rays_o=ray_positions,
        rays_d=ray_directions,
        alpha_fn=utils.compose(opacity_evaluator, operator.itemgetter(0)),
        render_step_size=marching_step_size,
        stratified=stratified,
    )

    sampled_opacities, sampled_gradients, *multi_sampled_features = opacity_evaluator(
        prev_sampled_distances=prev_ray_distances,
        next_sampled_distances=next_ray_distances,
        ray_indices=ray_indices,
    )

    # NOTE: NerfAcc low-level API
    # NerfAcc flattens the ray dimension and sample dimension because the number of samples for each ray is different.
    # So the per-ray cumulative products cannot be computed parallelly with only pure PyTorch API.
    # Therefore, only for this part, I use NerfAcc's low-level API, `nerfacc.exclusive_prod`.
    # The reason I don't write the entire part of rendering with NerfAcc's high-level API is to accommodate a variety of cases,
    # especially when we want to regularize the gradients of the SDF.
    packed_info = nerfacc.pack_info(ray_indices, ray_positions.shape[0])
    accumulated_transmittances = nerfacc.exclusive_prod(1.0 - sampled_opacities, packed_info)

    sampled_weights = accumulated_transmittances * sampled_opacities
    sampled_weights = sampled_weights.unsqueeze(-1)

    multi_accumulated_features = [
        torch.index_add(
            input=sampled_features.new_zeros(
                ray_positions.shape[0],
                *sampled_features.shape[1:],
            ),
            index=ray_indices,
            source=sampled_features * sampled_weights,
            dim=0,
        )
        for sampled_features in multi_sampled_features
    ]

    return *multi_accumulated_features, sampled_gradients
