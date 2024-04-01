import torch
import torch.nn as nn
from deepinv.loss.loss import Loss


class SupLoss(Loss):
    r"""
    Standard supervised loss

    The supervised loss is defined as

    .. math::

        \|x-\inverse{y}\|^2

    where :math:`\inverse{y}` is the reconstructed signal and :math:`x` is the ground truth target.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    """

    def __init__(self, metric=torch.nn.MSELoss()):
        super(SupLoss, self).__init__()
        self.name = "supervised"
        self.metric = metric

    def forward(self, x, x_net, **kwargs):
        r"""
        Computes the loss.

        :param torch.Tensor x: Target (ground-truth) image.
        :param torch.Tensor x_net: Reconstructed image :math:\inverse{y}.
        :return: (torch.Tensor) loss.
        """
        return self.metric(x_net, x)
