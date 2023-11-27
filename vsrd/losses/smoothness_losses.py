import torch
import torch.nn as nn

from . import utils


def gradient_x(inputs, padding=(0, 1), padding_mode="replicate"):
    inputs = nn.functional.pad(inputs, (*padding, 0, 0), padding_mode)
    gradients = inputs[..., :, 1:] - inputs[..., :, :-1]
    return gradients


def gradient_y(inputs, padding=(0, 1), padding_mode="replicate"):
    inputs = nn.functional.pad(inputs, (0, 0, *padding), padding_mode)
    gradients = inputs[..., 1:, :] - inputs[..., :-1, :]
    return gradients


@utils.reduced
def smoothness_loss(inputs, references, normalize=True, epsilon=1e-6):

    if normalize:
        means = torch.mean(inputs, dim=(-2, -1), keepdim=True)
        inputs = inputs / (means + epsilon)

    input_gradients_x = torch.abs(gradient_x(inputs))
    input_gradients_y = torch.abs(gradient_y(inputs))

    reference_gradients_x = torch.abs(gradient_x(references))
    reference_gradients_y = torch.abs(gradient_y(references))

    gradient_weights_x = torch.exp(-torch.mean(reference_gradients_x, dim=1, keepdim=True))
    gradient_weights_y = torch.exp(-torch.mean(reference_gradients_y, dim=1, keepdim=True))

    input_gradients_x = input_gradients_x * gradient_weights_x
    input_gradients_y = input_gradients_y * gradient_weights_y

    smoothness_losses = input_gradients_x + input_gradients_y

    return smoothness_losses


@utils.reduced
def motion_smoothness_loss(inputs, epsilon=1e-6):
    gradients_x = torch.abs(gradient_x(inputs))
    gradients_y = torch.abs(gradient_y(inputs))
    smoothness_losses = torch.sqrt(gradients_x ** 2.0 + gradients_y ** 2.0 + epsilon)
    return smoothness_losses


@utils.reduced
def motion_sparsity_loss(inputs, epsilon=1e-6):
    with torch.no_grad():
        means = torch.mean(torch.abs(inputs), dim=(-2, -1), keepdim=True)
    sparsity_losses = torch.sqrt(torch.abs(inputs) * means + means * means + epsilon)
    return sparsity_losses
