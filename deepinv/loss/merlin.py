from __future__ import annotations
from typing import Union
import torch
import math
import warnings
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric, MSE
from deepinv.models.base import Reconstructor


class MerlinLoss(Loss):
    r""" 
    Merlin splitting loss

    Splits the complex measurement into real and imaginary parts to compute
    the self-supervised loss:
    
    This loss is used for SAR despeckling in MERLIN :footcite:t:`dalsasso2021merlin`.
    """

    def __init__(self, metric: Metric = MSE, eps: float = 1e-10):
        r"""
        :param metric: Metric to be used. Can be a string (see :class:`deepinv.loss.metric.Metric`) or an instance of :class:`deepinv.loss.metric.Metric`.
        :param eps: Small constant to avoid division by zero.
        """
        super(MerlinLoss, self).__init__()
        self.metric = metric

    def adapt_model(self, model, **kwargs):
        return MerlinModel(model)

    def forward(self, x_net, x, y, physics, model, **kwargs):
        r"""
        Computes the MERLIN Loss.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor x: Reference image.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.

        :return: (:class:`torch.Tensor`) loss, the tensor size might be (1,) or (batch size,).
        """
        # Compute the residual
        target = y.real
        return self.metric(x_net, target)


class MerlinModel(Reconstructor):
    r"""
    MERLIN model wrapper.
    """

    def __init__(self, model: torch.nn.Module):
        super(MerlinModel, self).__init__()
        self.model = model

    def forward(self, y, physics, update_parameters=False, **kwargs):
        if self.model.training:
            y_noisy = y.imag
            return self.model(y_noisy, physics, **kwargs)
        else:
            x_hat_real = self.model(y.real, physics, **kwargs)
            x_hat_imag = self.model(y.imag, physics, **kwargs)
            return (x_hat_real + x_hat_imag) / 2
