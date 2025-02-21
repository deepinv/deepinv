from typing import Union

import torch
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.transform.base import Transform


class MCLoss(Loss):
    r"""
    Measurement consistency loss

    This loss enforces that the reconstructions are measurement-consistent, i.e., :math:`y=\forw{\inverse{y}}`.

    The measurement consistency loss is defined as

    .. math::

        \|y-\forw{\inverse{y}}\|^2

    where :math:`\inverse{y}` is the reconstructed signal and :math:`A` is a forward operator.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    """

    def __init__(self, metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss()):
        super(MCLoss, self).__init__()
        self.name = "mc"
        self.metric = metric

    def forward(self, y, x_net, physics, **kwargs):
        r"""
        Computes the measurement splitting loss

        :param torch.Tensor y: measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: forward operator associated with the measurements.
        :return: (:class:`torch.Tensor`) loss.
        """
        return self.metric(physics.A(x_net), y)
