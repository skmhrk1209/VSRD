import torch
import torch.nn as nn


def quadrature_sampler(bins, deterministic=False):
    weights = 0.5 if deterministic else torch.rand_like(bins[..., :-1])
    samples = torch.lerp(bins[..., :-1], bins[..., 1:], weights)
    return samples


def inverse_transform_sampler(bins, weights, num_samples, deterministic=False):

    pdf = nn.functional.normalize(weights, p=1, dim=-1)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = nn.functional.pad(cdf, (1, 0))

    if deterministic:
        uniform = torch.linspace(0.0, 1.0, num_samples, device=cdf.device)
        uniform = uniform.expand(*cdf.shape[:-1], -1)
    else:
        uniform = torch.rand(*cdf.shape[:-1], num_samples, device=cdf.device)
        uniform = torch.sort(uniform, dim=-1, descending=False).values

    indices = torch.searchsorted(cdf, uniform, right=False)
    indices = torch.clamp(indices, min=1, max=cdf.shape[-1] - 1)

    min_cdf = torch.gather(cdf, index=indices - 1, dim=-1)
    max_cdf = torch.gather(cdf, index=indices, dim=-1)

    min_bins = torch.gather(bins, index=indices - 1, dim=-1)
    max_bins = torch.gather(bins, index=indices, dim=-1)

    weights = (uniform - min_cdf) / (max_cdf - min_cdf + 1e-6)
    samples = torch.lerp(min_bins, max_bins, weights)

    return samples
