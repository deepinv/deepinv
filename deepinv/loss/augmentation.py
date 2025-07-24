from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.physics.mri import MRI
from deepinv.transform.base import Transform, Identity
from deepinv.transform.rotate import Rotate
from deepinv.transform.shift import Shift


class AugmentConsistencyLoss(Loss):
    r"""Data augmentation consistency (DAC) loss.

    Performs data augmentation in measurement domain as proposed by :footcite:t:`desai2021vortex`.

    The loss is defined as follows:

    :math:`\mathcal{L}(T_e\inverse{y,A},\inverse{T_i y,A T_e^{-1}})`

    where :math:`T_i` is a :class:`deepinv.transform.Transform` for which we should learn an invariant mapping,
    and :math:`T_e` is a :class:`deepinv.transform.Transform` for which we should learn an equivariant mapping.

    .. note::

        If :math:`T_e` is specified, the mapping is performed in the image domain and the model is assumed to take :math:`A^\top y` as input.

    By default, for :math:`T_i` we add random noise :class:`deepinv.transform.RandomNoise` and random phase error :class:`deepinv.transform.RandomPhaseError`.
    By default, for :math:`T_e` we use random shift :class:`deepinv.transform.Shift` and random rotates :class:`deepinv.transform.Rotate`.

    .. note::

        See :ref:`transform` for a guide on all available transforms, and how to compose them. For example, you can easily
        compose further transforms such as  ``Rotate(rng=rng, multiples=90) | Scale(factors=[0.75, 1.25], rng=rng) | Reflect(rng=rng)``.

    :param deepinv.transform.Transform T_i: invariant transform performed on :math:`y`.
    :param deepinv.transform.Transform T_e: equivariant transform performed on :math:`A^\top y`.
    :param deepinv.loss.metric.Metric, torch.nn.Module metric: metric for calculating loss.
    :param bool no_grad: if ``True``, only propagate gradients through augmented branch as per original paper,
        if ``False``, propagate through both branches.
    :param torch.Generator rng: torch random number generator to pass to transforms.


    """

    def __init__(
        self,
        T_i: Transform = None,
        T_e: Transform = None,
        metric: Union[Metric, nn.Module] = torch.nn.MSELoss(),
        no_grad: bool = True,
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.T_i = T_i if T_i is not None else Identity()
        self.T_e = (
            T_e
            if T_e is not None
            else Shift(shift_max=0.1, rng=rng) | Rotate(rng=rng, limits=15)
        )
        self.no_grad = no_grad

    def forward(self, x_net: Tensor, y: Tensor, physics: MRI, model, **kwargs):
        r"""
        Data augmentation consistency loss forward pass.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.

        :return: (:class:`torch.Tensor`) loss, the tensor size might be (1,) or (batch size,).
        """
        if self.no_grad:
            x_net = x_net.detach()

        # Sample image transform
        e_params = self.T_e.get_params(x_net)

        # Augment input
        x_aug = self.T_e(physics.A_adjoint(self.T_i(y)), **e_params)

        # Transform physics
        physics2 = physics.clone()
        A, A_adjoint, A_dagger = physics2.A, physics2.A_adjoint, physics2.A_dagger
        physics2.A = lambda x, *args, **kwargs: A(
            self.T_e.inverse(x, **e_params), *args, **kwargs
        )
        physics2.A_adjoint = lambda y, *args, **kwargs: self.T_e(
            A_adjoint(y, *args, **kwargs), **e_params
        )
        physics2.A_dagger = lambda y, *args, **kwargs: self.T_e(
            A_dagger(y, *args, **kwargs), **e_params
        )

        # Pass through network
        x_aug_net = model(physics2.A(x_aug), physics2)

        return self.metric(self.T_e(x_net, **e_params), x_aug_net)
