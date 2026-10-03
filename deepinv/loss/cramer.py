from __future__ import annotations

import torch
import torch.nn as nn

from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.models.noise_level_estimation import PatchCovarianceNoiseEstimator
from deepinv.models.anscombe import generalized_anscombe_transform


def estimate_noise(image, patch_size=8, stride=3, estimator=None):
    if isinstance(estimator, PatchCovarianceNoiseEstimator):
        return estimator.estimate_noise(image, patch_size=patch_size, stride=stride)
    return estimator(image)


class CramerGaussianLoss(Loss):
    r"""
    Cramér--Gaussian loss for blind Poisson--Gaussian noise estimation.

    Given a noisy measurement :math:`y`, a model predicts the Gaussian standard
    deviation :math:`\hat{\sigma}` and Poisson gain :math:`\hat{\gamma}`. The
    normalized generalized Anscombe transform is

    .. math::

        z = \frac{2}{\hat{\gamma}}\sqrt{\hat{\gamma}y
        + \frac{3}{8}\hat{\gamma}^2 + \hat{\sigma}^2}.

    The loss encourages :math:`z` to have unit noise standard deviation:

    .. math::

        \mathcal{L}(y) = d\left(E(z), 1\right),

    where :math:`E` is a Gaussian noise estimator and :math:`d` is the selected
    metric. The model output ``x_net`` must be a dictionary containing
    ``"sigma"`` and ``"gain"``.

    The loss was first introduced in :footcite:t:`byun2021fbi` for blind Poisson--Gaussian noise parameter estimation.

    :param Metric, torch.nn.Module metric: Metric used to
        compare the estimated standard deviation with one. By default, uses MSE.
    :param torch.nn.Module gaussian_estimator: Differentiable Gaussian noise
        estimator. By default, uses
        :class:`deepinv.models.PatchCovarianceNoiseEstimator`.
    :param int patch_size: Patch size used by the patch-covariance estimator.
        Default: 8.
    :param int stride: Patch stride used by the patch-covariance estimator.
        Default: 3.
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

        if gaussian_estimator is None:
            self.gaussian_estimator = PatchCovarianceNoiseEstimator()
        else:
            self.gaussian_estimator = gaussian_estimator

    def forward(self, x_net, y, **kwargs):
        r"""Compute the Cramér--Gaussian loss.

        :param dict[str, torch.Tensor] x_net: Estimated ``sigma`` and ``gain``.
        :param torch.Tensor y: Noisy Poisson--Gaussian measurements.
        :return: (:class:`torch.Tensor`) Cramér--Gaussian loss.
        """

        sigma, gain = x_net["sigma"], x_net["gain"]
        gat_est = generalized_anscombe_transform(
            y, gain=gain, sigma=sigma, normalize=True
        )
        std_est = estimate_noise(
            gat_est, self.psize, self.stride, self.gaussian_estimator
        )
        return self.metric(std_est, torch.ones_like(std_est))
