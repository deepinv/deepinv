from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.physics.mri import MRI
from deepinv.physics.forward import LinearPhysics
from deepinv.physics.inpainting import Inpainting
from deepinv.transform.base import Transform
from deepinv.transform.rotate import Rotate
from deepinv.transform.shift import Shift
from deepinv.transform.augmentation import RandomPhaseError, RandomNoise


class VORTEXLoss(Loss):
    r"""Measurement data augmentation loss.

    Performs data augmentation in measurement domain as proposed by
    `VORTEX: Physics-Driven Data Augmentations Using Consistency Training for Robust Accelerated MRI Reconstruction <https://arxiv.org/abs/2111.02549>`_.

    The loss is defined as follows:

    :math:`\mathcal{L}(T_e\inverse{y,A},\inverse{T_i y,A T_e^{-1}})`

    where :math:`T_1` is a random :class:`deepinv.transform.Transform` defined in k-space and
    :math:`T_2` is a random :class:`deepinv.transform.Transform` defined in image space.
    By default, for :math:`T_1` we add random noise :class:`deepinv.transform.RandomNoise` and random phase error :class:`deepinv.transform.RandomPhaseError`.
    By default, for :math:`T_2` we use random shift :class:`deepinv.transform.Shift` and random rotates :class:`deepinv.transform.Rotate`.

    .. note::

        See :ref:`transform` for a guide on all available transforms, and how to compose them. For example, you can easily
        compose further transforms such as  ``Rotate(rng=rng, multiples=90) | Scale(factors=[0.75, 1.25], rng=rng) | Reflect(rng=rng)``.

    .. note::

        For now, this loss is only available for MRI and Inpainting problems, but it is easily generalisable to other problems.

    :param deepinv.transform.Transform T_1: k-space transform.
    :param deepinv.transform.Transform T_2: image transform.
    :param deepinv.loss.metric.Metric, torch.nn.Module metric: metric for calculating loss.
    :param bool no_grad: if ``True``, only propagate gradients through augmented branch as per original paper,
        if ``False``, propagate through both branches.
    :param torch.Generator rng: torch random number generator to pass to transforms.
    """

    def __init__(
        self,
        T_1: Transform = None,
        T_2: Transform = None,
        metric: Union[Metric, nn.Module] = torch.nn.MSELoss(),
        no_grad: bool = True,
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.T_1 = (
            T_1
            if T_1 is not None
            else RandomPhaseError(scale=0.1, rng=rng) * RandomNoise(rng=rng)
        )
        self.T_2 = (
            T_2
            if T_2 is not None
            else Shift(shift_max=0.1, rng=rng) | Rotate(rng=rng, limits=15)
        )
        self.no_grad = no_grad

    class TransformedPhysics(LinearPhysics):
        """Pre-multiply physics with transform.

        :param deepinv.physics.Physics: original physics (only supports MRI and Inpainting for now)
        :param deepinv.transform.Transform: transform object
        :param dict transform_params: fixed parameters for deterministic transform.
        """

        def __init__(
            self,
            physics: Union[MRI, Inpainting],
            transform: Transform,
            transform_params: dict,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                img_size=getattr(physics, "img_size", None),
                tensor_size=getattr(physics, "tensor_size", None),
                mask=getattr(physics, "mask", None),
                device=getattr(physics, "device", None),
                three_d=getattr(physics, "three_d", None),
                **kwargs,
            )
            self.transform = transform
            self.transform_params = transform_params

        def A(self, x, *args, **kwargs):
            return super().A(
                self.transform.inverse(x, **self.transform_params), *args, **kwargs
            )

        def A_adjoint(self, y, *args, **kwargs):
            return self.transform(
                super().A_adjoint(y, *args, **kwargs), **self.transform_params
            )

    def forward(self, x_net: Tensor, y: Tensor, physics: MRI, model, **kwargs):
        r"""
        VORTEX loss forward pass.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.

        :return: (:class:`torch.Tensor`) loss, the tensor size might be (1,) or (batch size,).
        """
        if self.no_grad:
            x_net = x_net.detach()

        # Sample image transform
        e_params = self.T_2.get_params(x_net)

        # Augment input
        x_aug = self.T_2(physics.A_adjoint(self.T_1(y)), **e_params)

        # Pass through network
        physics2 = self.TransformedPhysics(physics, self.T_2, e_params)
        x_aug_net = model(physics2(x_aug), physics2)

        return self.metric(self.T_2(x_net, **e_params), x_aug_net)
