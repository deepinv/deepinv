from typing import Union, Optional
from copy import deepcopy

import numpy as np
import torch

from deepinv.loss.loss import Loss
from deepinv.loss.ei import EILoss
from deepinv.loss.metric.metric import Metric
from deepinv.physics import Physics
from deepinv.physics.generator import PhysicsGenerator
from deepinv.transform.base import Transform


class MOILoss(Loss):
    r"""
    Multi-operator imaging loss

    This loss can be used to learn when signals are observed via multiple (possibly incomplete)
    forward operators :math:`\{A_g\}_{g=1}^{G}`,
    i.e., :math:`y_i = A_{g_i}x_i` where :math:`g_i\in \{1,\dots,G\}` (see https://arxiv.org/abs/2201.12151).


    The measurement consistency loss is defined as

    .. math::

        \| \hat{x} - \inverse{A_g\hat{x},A_g} \|^2

    where :math:`\hat{x}=\inverse{y,A_s}` is a reconstructed signal (observed via operator :math:`A_s`) and
    :math:`A_g` is a forward operator sampled at random from a set :math:`\{A_g\}_{g=1}^{G}`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    The operators can be passed as a list of physics or as a single physics with a random physics generator.

    :param list[Physics], Physics physics: list of physics containing the :math:`G` different forward operators
            associated with the measurements, or single physics, or None. If single physics or None, physics generator must be used.
            If None, physics taken during ``forward``.
    :param PhysicsGenerator physics_generator: random physics generator that generates new params, if physics is not a list.
    :param Metric, torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float weight: total weight of the loss
    :param bool apply_noise: if ``True``, the augmented measurement is computed with the full sensing model
        :math:`\sensor{\noise{\forw{\hat{x}}}}` (i.e., noise and sensor model),
        otherwise is generated as :math:`\forw{\hat{x}}`.
    :param torch.Generator rng: torch randon number generator for randomly selecting from physics list. If using physics generator, rng is ignored.
    """

    def __init__(
        self,
        physics: Optional[Union[list[Physics], Physics]] = None,
        physics_generator: Optional[PhysicsGenerator] = None,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        apply_noise: bool = True,
        weight: float = 1.0,
        rng: Optional[torch.Generator] = None,
        *args,
        **kwargs,
    ):
        super(MOILoss, self).__init__(*args, **kwargs)
        self.name = "moi"
        self.physics = physics
        self.physics_generator = physics_generator
        self.metric = metric
        self.weight = weight
        self.noise = apply_noise
        self.rng = rng if rng is not None else torch.Generator()

        if isinstance(self.physics, (list, tuple)):
            if self.physics_generator is not None:
                raise ValueError(
                    "physics_generator cannot be used if a list of physics is used."
                )
        else:
            if self.physics_generator is None:
                raise ValueError(
                    "physics_generator must be passed if single physics is used or is None."
                )

    def next_physics(self, physics):
        """Create random physics.

        If physics is a list, select one at random. If physics generator is to be used, generate a new set of params at random.

        :param Physics physics: forward physics. If None, use physics passed at init.
        """
        if self.physics_generator is None:
            j = torch.randint(0, len(self.physics), (1,), generator=self.rng).item()
            physics_cur = self.physics[j]
        else:
            physics_cur = deepcopy(
                self.physics if self.physics is not None else physics
            )
            params = self.physics_generator.step()
            physics_cur.update_parameters(**params)
        return physics_cur

    def forward(self, x_net, physics, model, **kwargs):
        r"""
        Computes the MOI loss.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param Physics physics: measurement physics.
        :param torch.nn.Module model: Reconstruction function.
        :return: (:class:`torch.Tensor`) loss.
        """
        physics_cur = self.next_physics(physics)

        if self.noise:
            y = physics_cur(x_net)
        else:
            y = physics_cur.A(x_net)

        x2 = model(y, physics_cur)

        return self.weight * self.metric(x2, x_net)


class MOEILoss(EILoss, MOILoss):
    r"""Multi-operator equivariant imaging.

    This loss extends the equivariant loss :class:`deepinv.loss.EILoss`, where the signals are not only
    assumed to be invariant to a group of transformations, but also observed
    via multiple (possibly incomplete) forward operators :math:`\{A_s\}_{s=1}^{S}`,
    i.e., :math:`y_i = A_{s_i}x_i` where :math:`s_i\in \{1,\dots,S\}`.

    The multi-operator equivariance loss is defined as

    .. math::

        \| T_g \hat{x} - \inverse{A_2 T_g \hat{x}, A_2}\|^2

    where :math:`\hat{x}=\inverse{y,A_1}` is a reconstructed signal (observed via operator :math:`A_1`),
    :math:`A_2` is a forward operator sampled at random from a set :math:`\{A_2\}_{s=1}^{S}` and
    :math:`T_g` is a transformation sampled at random from a group :math:`g\sim\group`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    The operators can be passed as a list of physics or as a single physics with a random physics generator.

    See :class:`deepinv.loss.EILoss` for all parameter details for EI.

    :param deepinv.transform.Transform transform: Transform to generate the virtually
        augmented measurement. It can be any torch-differentiable function (e.g., a ``torch.nn.Module``).
    :param list[Physics], Physics physics: list of physics containing the :math:`G` different forward operators
            associated with the measurements, or single physics, or None. If single physics or None, physics generator must be used.
            If None, physics taken during ``forward``.
    :param PhysicsGenerator physics_generator: random physics generator that generates new params, if physics is not a list.
    :param Metric, torch.nn.Module metric: Metric used to compute the error between the reconstructed augmented measurement and the reference
        image.
    :param bool apply_noise: if ``True``, the augmented measurement is computed with the full sensing model
        :math:`\sensor{\noise{\forw{\hat{x}}}}` (i.e., noise and sensor model),
        otherwise is generated as :math:`\forw{\hat{x}}`.
    :param float weight: Weight of the loss.
    :param bool no_grad: if ``True``, the gradient does not propagate through :math:`T_g`. Default: ``False``.
        This option is useful for super-resolution problems, see https://arxiv.org/abs/2312.11232.
    :param torch.Generator rng: torch randon number generator for randomly selecting from physics list. If using physics generator, rng is ignored.
    """

    def __init__(
        self,
        transform: Transform,
        physics: Optional[Union[list[Physics], Physics]] = None,
        physics_generator: PhysicsGenerator = None,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        apply_noise: bool = True,
        weight: float = 1.0,
        no_grad: bool = False,
        rng: Optional[torch.Generator] = None,
    ):
        super().__init__(
            transform=transform,
            metric=metric,
            apply_noise=apply_noise,
            weight=weight,
            no_grad=no_grad,
            physics=physics,
            physics_generator=physics_generator,
            rng=rng,
        )
        self.name = "moei"

    def forward(self, x_net, physics, model, **kwargs):
        r"""
        Computes the MO-EI loss

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (:class:`torch.Tensor`) loss.
        """
        physics_cur = self.next_physics(physics)
        return EILoss.forward(self, x_net, physics_cur, model, **kwargs)
