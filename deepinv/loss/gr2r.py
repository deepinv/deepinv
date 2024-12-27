from __future__ import annotations
from typing import Union
import torch
import math
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.physics.noise import GaussianNoise, PoissonNoise, GammaNoise
from deepinv.physics.noise import NoiseModel


class GeneralizedR2RLoss(Loss):
    r"""
    Generalized Recorrupted-to-Recorrupted (GR2R) Loss

    """

    def __init__(
        self,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        noise_model: NoiseModel = GaussianNoise(0.1),
        alpha=0.5,
        eval_n_samples=5,
    ):
        super(GeneralizedR2RLoss, self).__init__()
        self.name = "gr2r"
        self.metric = metric
        self.alpha = alpha
        self.eval_n_samples = eval_n_samples
        self.noise_model = noise_model

    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Computes the GR2R Loss.

        """

        y1 = model.get_corruption()
        y2 = (1 / self.alpha) * (y - y1 * (1 - self.alpha))
        return self.metric(physics.A(x_net), y2)

    def adapt_model(self, model, **kwargs):
        return GeneralizedR2RModel(
            model, self.noise_model, self.alpha, self.eval_n_samples, **kwargs
        )


def set_gaussian_corruptor(y, alpha, sigma):
    mu = torch.ones_like(y) * 0.0
    sigma = torch.ones_like(y) * sigma
    sampler = torch.distributions.Normal(mu, sigma)
    corruptor = lambda: y + sampler.sample() * (math.sqrt(alpha / (1 - alpha)))
    return corruptor


def set_binomial_corruptor(y, alpha, gamma):
    z = y / gamma
    sampler = torch.distributions.Binomial(torch.round(z), alpha)
    corruptor = lambda: gamma * (z - sampler.sample()) / (1 - alpha)
    return corruptor


def set_beta_corruptor(y, alpha, l):
    tmp = torch.ones_like(y)
    concentration1 = tmp * l * alpha
    concentration0 = tmp * l * (1 - alpha)
    sampler = torch.distributions.Beta(concentration1, concentration0)
    corruptor = lambda: y * (1 - sampler.sample()) / (1 - alpha)
    return corruptor


class GeneralizedR2RModel(torch.nn.Module):
    r"""
    Generalized Recorrupted-to-Recorrupted (GR2R) Model

    """

    def __init__(self, model, noise_model, alpha, eval_n_samples):
        super(GeneralizedR2RModel, self).__init__()
        self.model = model
        self.noise_model = noise_model
        self.eval_n_samples = eval_n_samples
        self.alpha = alpha

    def forward(self, y, physics, update_parameters=False):

        eval_n_samples = 1 if self.training else self.eval_n_samples
        out = 0
        corruptor = self.get_corruptor(y)

        with torch.set_grad_enabled(self.training):
            for i in range(eval_n_samples):
                y1 = corruptor()
                out += self.model(y1, physics)

            if self.training and update_parameters:
                self.corruption = y1

            out = out / eval_n_samples
        return out

    def get_corruptor(self, y):
        alpha = self.alpha

        if isinstance(self.noise_model, GaussianNoise):

            sigma = self.noise_model.sigma
            return set_gaussian_corruptor(y, alpha, sigma)

        elif isinstance(self.noise_model, PoissonNoise):

            gain = self.noise_model.gain
            return set_binomial_corruptor(y, alpha, gain)

        elif isinstance(self.noise_model, GammaNoise):

            l = self.noise_model.l
            return set_beta_corruptor(y, alpha, l)

        else:
            raise ValueError(
                f"Noise model {self.noise_model} not supported, available options are Gaussian, Poisson and Gamma."
            )

    def get_corruption(self):
        return self.corruption
