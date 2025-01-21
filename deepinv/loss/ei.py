from typing import Union

import torch
import torch.nn as nn
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.transform.base import Transform


class EILoss(Loss):
    r"""
    Equivariant imaging self-supervised loss.

    Assumes that the set of signals is invariant to a group of transformations (rotations, translations, etc.)
    in order to learn from incomplete measurement data alone https://https://arxiv.org/pdf/2103.14756.pdf.

    The EI loss is defined as

    .. math::

        \| T_g \hat{x} - \inverse{\forw{T_g \hat{x}}}\|^2


    where :math:`\hat{x}=\inverse{y}` is a reconstructed signal and
    :math:`T_g` is a transformation sampled at random from a group :math:`g\sim\group`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param deepinv.transform.Transform transform: Transform to generate the virtually augmented measurement.
        It can be any torch-differentiable function (e.g., a :class:`torch.nn.Module`)
        including `torchvision transforms <https://pytorch.org/vision/stable/transforms.html>`_.
    :param Metric, torch.nn.Module metric: Metric used to compute the error between the reconstructed augmented measurement and the reference
        image.
    :param bool apply_noise: if ``True``, the augmented measurement is computed with the full sensing model
        :math:`\sensor{\noise{\forw{\hat{x}}}}` (i.e., noise and sensor model),
        otherwise is generated as :math:`\forw{\hat{x}}`.
    :param float weight: Weight of the loss.
    :param bool no_grad: if ``True``, the gradient does not propagate through :math:`T_g`. Default: ``False``.
        This option is useful for super-resolution problems, see https://arxiv.org/abs/2312.11232.
    """

    def __init__(
        self,
        transform: Transform,
        metric: Union[Metric, nn.Module] = torch.nn.MSELoss(),
        apply_noise=True,
        weight=1.0,
        no_grad=False,
        *args,
        **kwargs,
    ):
        super(EILoss, self).__init__(*args, **kwargs)
        self.name = "ei"
        self.metric = metric
        self.weight = weight
        self.T = transform
        self.noise = apply_noise
        self.no_grad = no_grad

    def forward(self, x_net, physics, model, **kwargs):
        r"""
        Computes the EI loss

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (:class:`torch.Tensor`) loss.
        """

        if self.no_grad:
            # NOTE: Calling both torch.no_grad() and detach() is not redundant.
            # One avoids unnecessary computations and makes the code more efficient
            # while the other ensures that x2 is marked as a leaf in the computational graph.
            with torch.no_grad():
                x2 = self.T(x_net)
                x2 = x2.detach()
        else:
            x2 = self.T(x_net)

        if self.noise:
            y = physics(x2)
        else:
            y = physics.A(x2)

        x3 = model(y, physics)

        loss_ei = self.weight * self.metric(x3, x2)
        return loss_ei
