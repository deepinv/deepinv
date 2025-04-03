from typing import Union

import torch
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric


class SupLoss(Loss):
    r"""
    Standard supervised loss

    The supervised loss is defined as

    .. math::

        \|x-\inverse{y}\|^2

    where :math:`\inverse{y}` is the reconstructed signal and :math:`x` is the ground truth target.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.
    If called with arguments ``x_net, x``, this is simply a wrapper for the metric ``metric``.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    """

    def __init__(self, metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss()):
        super().__init__()
        self.name = "supervised"
        self.metric = metric

    def forward(self, x_net, x, **kwargs):
        r"""
        Computes the loss.

        :param torch.Tensor x_net: Reconstructed image :math:\inverse{y}.
        :param torch.Tensor x: Target (ground-truth) image.
        :return: (:class:`torch.Tensor`) loss.
        """
        return self.metric(x_net, x)
