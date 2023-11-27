import torch
import torch.nn as nn

from . import utils
from . import classification_losses


@utils.reduced
def gaussian_nll(means, variances, targets, epsilon=1e-6):
    gaussian = torch.distributions.Normal(means, torch.sqrt(variances + epsilon))
    losses = -gaussian.log_prob(targets)
    return losses


@utils.reduced
def student_nll(means, shapes, scales, targets, epsilon=1e-6):
    """
    Negative Log-Likelihood (NLL) of Gaussian marginalized w.r.t. `variance`, which follows Inverse-Gamma distribution parametrized by `shape` and `scale`.
    This equals to NLL of generalized Student-t distribution parametrized by `dof=2*shape`, `loc=mean`, and `scale^2=scale/shape`.

    References:
        - [Reliable training and estimation of variance networks](https://arxiv.org/pdf/1906.03260.pdf)
            See Section 3.4 for the details. Other tequniques for variance estimation have also been proposed.
        - [Inverse-Gamma distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
        - [Gaussian-Inverse-Gamma distribution](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution)
        - [Generalized Student-t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution)

    Arguments:
        means (torch.Tensor): The `mean` parameter of Gaussian distribution.
        shapes (torch.Tensor): The `shape` parameter of Inverse-Gamma distribution.
            This should be positive, so it is recommended to apply `softplus` as the output activation function.
            Note that `shape > 1` is actually needed for computation of the expectation of the variance which follows Inverse-Gamma distribution.
        scales (torch.Tensor): The `scale` parameter of Inverse-Gamma distribution.
            This should be positive, so it is recommended to apply `softplus` as the output activation function.
        targets (torch.Tensor): The target variable.
    """
    degrees = 2.0 * shapes
    variances = scales / shapes
    student_t = torch.distributions.StudentT(degrees, means, torch.sqrt(variances + epsilon))
    losses = -student_t.log_prob(targets)
    return losses


@utils.reduced
def gaussian_energy_score(means, variances, targets, num_samples=1000, epsilon=1e-6):
    gaussian = torch.distributions.Normal(means, torch.sqrt(variances + epsilon))
    mc_samples = gaussian.rsample([num_samples]).to(targets)
    inter_losses = torch.mean(nn.functional.l1_loss(mc_samples, targets[None, ...], reduction="none"), dim=0)
    intra_losses = torch.mean(nn.functional.l1_loss(mc_samples[:-1, ...], mc_samples[1:, ...], reduction="none"), dim=0)
    losses = inter_losses - intra_losses * 0.5
    return losses


@utils.reduced
def student_energy_score(means, shapes, scales, targets, num_samples=1000, epsilon=1e-6):
    """
    Negative Log-Likelihood (NLL) of Gaussian marginalized w.r.t. `variance`, which follows Inverse-Gamma distribution parametrized by `shape` and `scale`.
    This equals to NLL of generalized Student-t distribution parametrized by `dof=2*shape`, `loc=mean`, and `scale^2=scale/shape`.

    References:
        - [Reliable training and estimation of variance networks](https://arxiv.org/pdf/1906.03260.pdf)
            See Section 3.4 for the details. Other tequniques for variance estimation have also been proposed.
        - [Inverse-Gamma distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
        - [Gaussian-Inverse-Gamma distribution](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution)
        - [Generalized Student-t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution)

    Arguments:
        means (torch.Tensor): The `mean` parameter of Gaussian distribution.
        shapes (torch.Tensor): The `shape` parameter of Inverse-Gamma distribution.
            This should be positive, so it is recommended to apply `softplus` as the output activation function.
            Note that `shape > 1` is actually needed for computation of the expectation of the variance which follows Inverse-Gamma distribution.
        scales (torch.Tensor): The `scale` parameter of Inverse-Gamma distribution.
            This should be positive, so it is recommended to apply `softplus` as the output activation function.
        targets (torch.Tensor): The target variable.
    """
    degrees = 2.0 * shapes
    variances = scales / shapes
    student_t = torch.distributions.StudentT(degrees, means, torch.sqrt(variances + epsilon))
    mc_samples = student_t.rsample([num_samples]).to(targets)
    inter_losses = torch.mean(nn.functional.l1_loss(mc_samples, targets[None, ...], reduction="none"), dim=0)
    intra_losses = torch.mean(nn.functional.l1_loss(mc_samples[:-1, ...], mc_samples[1:, ...], reduction="none"), dim=0)
    losses = inter_losses - intra_losses * 0.5
    return losses


@utils.reduced
def logit_gaussian_nll(means, variances, targets, epsilon=1e-6):
    logit_gaussian = torch.distributions.TransformedDistribution(
        base_distribution=torch.distributions.Normal(means, torch.sqrt(variances + epsilon)),
        transforms=torch.distributions.SigmoidTransform(),
    )
    losses = -logit_gaussian.log_prob(targets)
    return losses


@utils.reduced
def logit_student_nll(means, shapes, scales, targets, epsilon=1e-6):
    """
    Negative Log-Likelihood (NLL) of Gaussian marginalized w.r.t. `variance`, which follows Inverse-Gamma distribution parametrized by `shape` and `scale`.
    This equals to NLL of generalized Student-t distribution parametrized by `dof=2*shape`, `loc=mean`, and `scale^2=scale/shape`.

    References:
        - [Reliable training and estimation of variance networks](https://arxiv.org/pdf/1906.03260.pdf)
            See Section 3.4 for the details. Other tequniques for variance estimation have also been proposed.
        - [Inverse-Gamma distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
        - [Gaussian-Inverse-Gamma distribution](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution)
        - [Generalized Student-t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution)

    Arguments:
        means (torch.Tensor): The `mean` parameter of Gaussian distribution.
        shapes (torch.Tensor): The `shape` parameter of Inverse-Gamma distribution.
            This should be positive, so it is recommended to apply `softplus` as the output activation function.
            Note that `shape > 1` is actually needed for computation of the expectation of the variance which follows Inverse-Gamma distribution.
        scales (torch.Tensor): The `scale` parameter of Inverse-Gamma distribution.
            This should be positive, so it is recommended to apply `softplus` as the output activation function.
        targets (torch.Tensor): The target variable.
    """
    degrees = 2.0 * shapes
    variances = scales / shapes
    logit_student_t = torch.distributions.TransformedDistribution(
        base_distribution=torch.distributions.StudentT(degrees, means, torch.sqrt(variances + epsilon)),
        transforms=torch.distributions.SigmoidTransform(),
    )
    losses = -logit_student_t.log_prob(targets)
    return losses


@utils.reduced
def logit_gaussian_energy_score(means, variances, targets, num_samples=1000, epsilon=1e-6):
    logit_gaussian = torch.distributions.TransformedDistribution(
        base_distribution=torch.distributions.Normal(means, torch.sqrt(variances + epsilon)),
        transforms=torch.distributions.SigmoidTransform(),
    )
    mc_samples = logit_gaussian.rsample([num_samples]).to(targets)
    inter_losses = torch.mean(classification_losses.binary_cross_entropy(mc_samples, targets[None, ...], reduction="none"), dim=0)
    intra_losses = torch.mean(classification_losses.binary_cross_entropy(mc_samples[:-1, ...], mc_samples[1:, ...], reduction="none"), dim=0)
    losses = inter_losses - intra_losses * 0.5
    return losses


@utils.reduced
def logit_student_energy_score(means, shapes, scales, targets, num_samples=1000, epsilon=1e-6):
    """
    Negative Log-Likelihood (NLL) of Gaussian marginalized w.r.t. `variance`, which follows Inverse-Gamma distribution parametrized by `shape` and `scale`.
    This equals to NLL of generalized Student-t distribution parametrized by `dof=2*shape`, `loc=mean`, and `scale^2=scale/shape`.

    References:
        - [Reliable training and estimation of variance networks](https://arxiv.org/pdf/1906.03260.pdf)
            See Section 3.4 for the details. Other tequniques for variance estimation have also been proposed.
        - [Inverse-Gamma distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
        - [Gaussian-Inverse-Gamma distribution](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution)
        - [Generalized Student-t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution)

    Arguments:
        means (torch.Tensor): The `mean` parameter of Gaussian distribution.
        shapes (torch.Tensor): The `shape` parameter of Inverse-Gamma distribution.
            This should be positive, so it is recommended to apply `softplus` as the output activation function.
            Note that `shape > 1` is actually needed for computation of the expectation of the variance which follows Inverse-Gamma distribution.
        scales (torch.Tensor): The `scale` parameter of Inverse-Gamma distribution.
            This should be positive, so it is recommended to apply `softplus` as the output activation function.
        targets (torch.Tensor): The target variable.
    """
    degrees = 2.0 * shapes
    variances = scales / shapes
    logit_student_t = torch.distributions.TransformedDistribution(
        base_distribution=torch.distributions.StudentT(degrees, means, torch.sqrt(variances + epsilon)),
        transforms=torch.distributions.SigmoidTransform(),
    )
    mc_samples = logit_student_t.rsample([num_samples]).to(targets)
    inter_losses = torch.mean(classification_losses.binary_cross_entropy(mc_samples, targets[None, ...], reduction="none"), dim=0)
    intra_losses = torch.mean(classification_losses.binary_cross_entropy(mc_samples[:-1, ...], mc_samples[1:, ...], reduction="none"), dim=0)
    losses = inter_losses - intra_losses * 0.5
    return losses
