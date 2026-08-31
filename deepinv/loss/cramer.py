from __future__ import annotations
import inspect

import torch
import torch.nn as nn

from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.models.noise_level_estimation import PatchCovarianceNoiseEstimator, WaveletNoiseEstimator
from deepinv.models.anscombe import generalized_anscombe_transform, inverse_generalized_anscombe_transform

# Change this one line to select the estimator used by every experiment.
NOISE_ESTIMATOR = WaveletNoiseEstimator


def estimate_noise(image, patch_size=8, stride=3, estimator=None):
    """Run the shared noise estimator, forwarding patch options when supported."""
    estimator = (NOISE_ESTIMATOR() if estimator is None else estimator)
    if isinstance(estimator, WaveletNoiseEstimator):
        return estimator(image)
    method = getattr(estimator, "estimate_noise", None)
    if method is not None and "patch_size" in inspect.signature(method).parameters:
        return method(image, patch_size=patch_size, stride=stride)
    return estimator(image)


class CramerGaussianLoss(Loss):
    r"""
    Cramer-Gaussian loss.
    """

    def __init__(
        self,
        metric: Metric = None,
        gaussian_estimator: nn.Module | None = None,
        patch_size: int = 8,
        stride: int = 3,
    ):

        if metric is None:
            metric = torch.nn.MSELoss()
        super(CramerGaussianLoss, self).__init__()
        self._name = "cramer_gaussian"
        self.metric = metric
        self.psize = patch_size
        self.stride = stride

        self.gaussian_estimator = (
            NOISE_ESTIMATOR() if gaussian_estimator is None else gaussian_estimator
        )
            

    def forward(self, p_net, y, **kwargs):

        sigma, gain = p_net["sigma"], p_net["gain"]
        gat_est = generalized_anscombe_transform(y, gain=gain, sigma=sigma)
        gat_est = gat_est / gain
        std_est = estimate_noise(
            gat_est, self.psize, self.stride, self.gaussian_estimator
        )
        return self.metric(std_est,  torch.ones_like(std_est) )
