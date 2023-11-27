import torch
import torch.nn as nn

from . import utils


@utils.reduced
def ssim_loss(inputs, targets, C1=0.01 ** 2, C2=0.03 ** 2, kernel_size=3, stride=1, padding=1, padding_mode="reflect"):

    x, y = inputs, targets

    x = nn.functional.pad(x, [padding] * 4, padding_mode)
    y = nn.functional.pad(y, [padding] * 4, padding_mode)

    mu_x = nn.functional.avg_pool2d(x, kernel_size, stride)
    mu_y = nn.functional.avg_pool2d(y, kernel_size, stride)

    sigma_xx = nn.functional.avg_pool2d(x * x, kernel_size, stride) - mu_x * mu_x
    sigma_yy = nn.functional.avg_pool2d(y * y, kernel_size, stride) - mu_y * mu_y
    sigma_xy = nn.functional.avg_pool2d(x * y, kernel_size, stride) - mu_x * mu_y

    luminance_comparisons = (2.0 * mu_x * mu_y + C1) / (mu_x * mu_x + mu_y * mu_y + C1)
    contrast_structure_comparisons = (2.0 * sigma_xy + C2) / (sigma_xx + sigma_yy + C2)

    ssim_losses = luminance_comparisons * contrast_structure_comparisons
    ssim_losses = torch.clamp((1.0 - ssim_losses) / 2.0, 0.0, 1.0)

    return ssim_losses


@utils.reduced
def photometric_loss(inputs, targets, alpha=0.75):
    ssim_losses = ssim_loss(inputs, targets, reduction="none")
    huber_losses = nn.functional.smooth_l1_loss(inputs, targets, reduction="none")
    photometric_losses = ssim_losses * alpha + huber_losses * (1.0 - alpha)
    return photometric_losses
