import enum
import functools

import torch
import torch.nn as nn
import torchvision
import numpy as np

from .. import utils


class FractalBrownianMotion(nn.Module):
    """ Improved Perlin Noise Algorithm

    References:
        - [Improving Noise](https://mrl.cs.nyu.edu/~perlin/paper445.pdf)
        - [Simplex Noise](https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf)
    """

    class HermiteCurve(enum.Enum):
        CUBIC = enum.auto()
        QUINTIC = enum.auto()

    def __init__(self, amplitude, resolution, persistence, lacunarity, num_octaves, hermite_curve=HermiteCurve.CUBIC):
        super().__init__()
        self.amplitude = torch.tensor(amplitude)
        self.resolution = torch.tensor(resolution)
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.num_octaves = num_octaves
        self.hermite_curve = hermite_curve

    @staticmethod
    def perlin_noise(image_size, amplitude, resolution, hermite_curve):

        remnant = image_size % resolution
        padding = (resolution - remnant) % resolution
        image_size = image_size + padding

        positions = tuple(map(functools.partial(torch.linspace, 0.0), resolution, image_size))
        positions = torch.frac(torch.stack(torch.meshgrid(*positions, indexing="ij"), dim=0))

        distances_00 = torch.stack([positions[0, ...] - 0, positions[1, ...] - 0], dim=0)
        distances_10 = torch.stack([positions[0, ...] - 1, positions[1, ...] - 0], dim=0)
        distances_01 = torch.stack([positions[0, ...] - 0, positions[1, ...] - 1], dim=0)
        distances_11 = torch.stack([positions[0, ...] - 1, positions[1, ...] - 1], dim=0)

        angles = 2.0 * np.pi * torch.rand(*(resolution + 1))
        gradients = torch.stack([torch.cos(angles), torch.sin(angles)], dim=0)

        assert torch.all(image_size % resolution == 0)
        num_repeats = image_size // resolution

        gradients = torch.repeat_interleave(gradients, num_repeats[-2], dim=-2)
        gradients = torch.repeat_interleave(gradients, num_repeats[-1], dim=-1)

        gradients_00 = gradients[..., :-num_repeats[-2], :-num_repeats[-1]]
        gradients_10 = gradients[..., num_repeats[-2]:, :-num_repeats[-1]]
        gradients_01 = gradients[..., :-num_repeats[-2], num_repeats[-1]:]
        gradients_11 = gradients[..., num_repeats[-2]:, num_repeats[-1]:]

        noises_00 = torch.sum(distances_00 * gradients_00, dim=0, keepdim=True)
        noises_10 = torch.sum(distances_10 * gradients_10, dim=0, keepdim=True)
        noises_01 = torch.sum(distances_01 * gradients_01, dim=0, keepdim=True)
        noises_11 = torch.sum(distances_11 * gradients_11, dim=0, keepdim=True)

        if hermite_curve is __class__.HermiteCurve.CUBIC:
            hermite_curve = lambda x: 3 * x ** 2 - 2 * x ** 3
        elif hermite_curve is __class__.HermiteCurve.QUINTIC:
            hermite_curve = lambda x: 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3

        y_weights, x_weights = torch.unbind(hermite_curve(positions), dim=0)

        noises_0 = torch.lerp(noises_00, noises_10, y_weights)
        noises_1 = torch.lerp(noises_01, noises_11, y_weights)
        noises = torch.lerp(noises_0, noises_1, x_weights)

        min_value = torch.min(noises)
        max_value = torch.max(noises)
        noises = (noises - min_value) / (max_value - min_value)
        noises = (noises * 2.0 - 1.0) * amplitude

        noises = noises[..., padding[-2]:, padding[-1]:]

        return noises

    def forward_impl(self, image_size):
        noises = sum(
            FractalBrownianMotion.perlin_noise(
                image_size=image_size,
                amplitude=self.amplitude * self.persistence ** octave,
                resolution=self.resolution * self.lacunarity ** octave,
                hermite_curve=self.hermite_curve,
            )
            for octave in range(self.num_octaves)
        )
        return noises

    def forward(self, image, **kwargs):
        noises = self.forward_impl(torch.tensor(image.shape[-2:]))
        image = image + noises * torch.rand(())
        image = torch.clamp(image, 0.0, 1.0)
        return dict(kwargs, image=image)


class FastFractalBrownianMotion(FractalBrownianMotion):

    def __init__(self, *args, downscale_factor, interp_mode="bilinear", **kwargs):
        super().__init__(*args, **kwargs)
        self.downscale_factor = downscale_factor
        self.interp_mode = interp_mode

    def forward(self, image, **kwargs):
        noises = self.forward_impl(torch.tensor(image.shape[-2:]) // self.downscale_factor)
        noises = utils.unvectorize(nn.functional.interpolate)(
            input=noises,
            size=image.shape[-2:],
            mode=self.interp_mode,
            align_corners=False,
        )
        image = image + noises * torch.rand(())
        image = torch.clamp(image, 0.0, 1.0)
        return dict(kwargs, image=image)


class RandomConvolution(nn.Module):

    def __init__(self, kernel_sizes, depthwise=False):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.depthwise = depthwise

    def forward(self, image, **kwargs):
        kernel_index = torch.randint(len(self.kernel_sizes), ())
        kernel_size = self.kernel_sizes[kernel_index.item()]
        groups = image.shape[-3] if self.depthwise else 1
        weight = torch.randn(
            image.shape[-3],
            image.shape[-3] // groups,
            kernel_size,
            kernel_size,
        )
        weight = weight / torch.sum(weight, dim=(-3, -2, -1), keepdim=True)
        convolved_image = nn.functional.conv2d(
            input=image.unsqueeze(0),
            weight=weight,
            padding=(kernel_size - 1) // 2,
            groups=groups,
        ).squeeze(0)
        image = torch.lerp(image, convolved_image, torch.rand(()))
        image = torch.clamp(image, 0.0, 1.0)
        return dict(kwargs, image=image)


class ColorJitter(torchvision.transforms.ColorJitter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brightness_sampler = torch.distributions.uniform.Uniform(*self.brightness)
        self.contrast_sampler = torch.distributions.uniform.Uniform(*self.contrast)
        self.saturation_sampler = torch.distributions.uniform.Uniform(*self.saturation)
        self.hue_sampler = torch.distributions.uniform.Uniform(*self.hue)
        self.update_params()

    def update_params(self):
        transforms = [
            torchvision.transforms.Lambda(functools.partial(
                torchvision.transforms.functional.adjust_brightness,
                brightness_factor=self.brightness_sampler.sample(),
            )),
            torchvision.transforms.Lambda(functools.partial(
                torchvision.transforms.functional.adjust_contrast,
                contrast_factor=self.contrast_sampler.sample(),
            )),
            torchvision.transforms.Lambda(functools.partial(
                torchvision.transforms.functional.adjust_saturation,
                saturation_factor=self.saturation_sampler.sample(),
            )),
            torchvision.transforms.Lambda(functools.partial(
                torchvision.transforms.functional.adjust_hue,
                hue_factor=self.hue_sampler.sample(),
            )),
        ]
        transforms = list(map(transforms.__getitem__, torch.randperm(4).tolist()))
        self.transform = torchvision.transforms.Compose(transforms)

    def forward(self, image, augmented_image=None, **kwargs):
        if augmented_image is None:
            augmented_image = image
        augmented_image = self.transform(augmented_image)
        return dict(kwargs, image=image, augmented_image=augmented_image)
