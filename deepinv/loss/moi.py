from typing import Union, Optional
from copy import deepcopy

import numpy as np
import torch

from deepinv.loss.loss import Loss
from deepinv.loss.ei import EILoss
from deepinv.physics import Physics
from deepinv.physics.generator import PhysicsGenerator


class MOILoss(Loss):
    r"""
    Multi-operator imaging loss

    This loss can be used to learn when signals are observed via multiple (possibly incomplete)
    forward operators :math:`\{A_g\}_{g=1}^{G}`,
    i.e., :math:`y_i = A_{g_i}x_i` where :math:`g_i\in \{1,\dots,G\}` (see https://arxiv.org/abs/2201.12151).


    The measurement consistency loss is defined as

    .. math::

        \| \hat{x} - \inverse{\hat{x},A_g} \|^2

    where :math:`\hat{x}=\inverse{y,A_s}` is a reconstructed signal (observed via operator :math:`A_s`) and
    :math:`A_g` is a forward operator sampled at random from a set :math:`\{A_g\}_{g=1}^{G}`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    The operators can be passed as a list of physics or as a single physics with a random physics generator.

    :param list[Physics], Physics physics: list of physics containing the :math:`G` different forward operators
            associated with the measurements, or single physics, or None. If single physics or None, physics generator must be used.
            If None, physics taken during ``forward``.
    :param PhysicsGenerator physics_generator: random physics generator that generates new params, if physics is not a list.
    :param torch.nn.Module metric: metric used for computing data consistency,
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
        metric=torch.nn.MSELoss(),
        apply_noise=True,
        weight=1.0,
        rng=torch.Generator(),
    ):
        super(MOILoss, self).__init__()
        self.name = "moi"
        self.physics = physics
        self.physics_generator = physics_generator
        self.metric = metric
        self.weight = weight
        self.noise = apply_noise
        self.rng = rng

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
        :return: (torch.Tensor) loss.
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

    See :class:`deepinv.loss.EILoss` for all parameter details for EI.

    :param list[Physics], Physics physics: list of physics containing the :math:`G` different forward operators
            associated with the measurements, or single physics, or None. If single physics or None, physics generator must be used.
            If None, physics taken during ``forward``.
    :param PhysicsGenerator physics_generator: random physics generator that generates new params, if physics is not a list.
    """

    def __init__(
        self,
        *args,
        physics: Optional[Union[list[Physics], Physics]] = None,
        physics_generator: PhysicsGenerator = None,
        **kwargs,
    ):
        EILoss.__init__(*args, **kwargs)
        self.physics = physics
        self.physics_generator = physics_generator

    def forward(self, x_net, physics, model, **kwargs):
        physics_cur = self.next_physics(physics)
        return EILoss.forward(x_net, physics_cur, model, **kwargs)
