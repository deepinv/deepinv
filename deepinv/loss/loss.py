from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics, StackedPhysics
    from deepinv.utils import TensorList


class Loss(torch.nn.Module):
    r"""
    Base class for all loss functions.

    Sets a template for the loss functions, whose forward method must follow the input parameters in
    :func:`deepinv.loss.Loss.forward`.
    """

    def __init__(self):
        super(Loss, self).__init__()

    def forward(
        self,
        x_net: Tensor,
        x: Tensor,
        y: Tensor,
        physics: Physics,
        model: Module,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the loss.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor x: Reference image.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.

        :return: (:class:`torch.Tensor`) loss, the tensor size might be (1,) or (batch size,).
        """
        raise NotImplementedError(
            "The method 'forward' must be implemented in the subclass."
        )

    def adapt_model(self, model: Module, **kwargs) -> torch.nn.Module:
        r"""
        Some loss functions require the model forward call to be adapted before the forward pass.

        :param torch.nn.Module model: reconstruction model
        """
        return model


class StackedPhysicsLoss(Loss):
    r"""
    Loss function for stacked physics operators.

    Adapted to :class:`deepinv.physics.StackedPhysics` physics composed of multiple physics operators.

    :param list[deepinv.loss.Loss] losses: list of loss functions for each physics operator.

    |sep|

    :Examples:

        Gaussian and Poisson losses function for a stacked physics operator:

        >>> import torch
        >>> import deepinv as dinv
        >>> # define two observations, one with Gaussian noise and one with Poisson noise
        >>> physics1 = dinv.physics.Denoising(dinv.physics.GaussianNoise(.1))
        >>> physics2 = dinv.physics.Denoising(dinv.physics.PoissonNoise(.1))
        >>> physics = dinv.physics.StackedLinearPhysics([physics1, physics2])
        >>> loss1 = dinv.loss.SureGaussianLoss(.1)
        >>> loss2 = dinv.loss.SurePoissonLoss(.1)
        >>> loss = dinv.loss.StackedPhysicsLoss([loss1, loss2])
        >>> x = torch.ones(1, 1, 5, 5) # image
        >>> y = physics(x) # noisy measurements
        >>> # define a denoiser model
        >>> model = dinv.models.ArtifactRemoval(dinv.models.MedianFilter(3))
        >>> x_net = model(y, physics)
        >>> l = loss(x_net, x, y, physics, model)
    """

    def __init__(self, losses):
        super(StackedPhysicsLoss, self).__init__()
        self.losses = losses

    def forward(
        self,
        x_net: Tensor,
        x: Tensor,
        y: TensorList,
        physics: StackedPhysics,
        model: Module,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the loss as

        .. math::

            \sum_i \mathcal{L}_i(x, y_i, \inverse{y}, \physics_i, \model),

        where :math:`i` is the index of the physics operator in the stacked physics.

        :param deepinv.utils.TensorList y: Measurement.
        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.

        """
        loss = 0
        for i, loss_fn in enumerate(self.losses):

            def model_aux(s, p, **kwargs):
                r = y.clone()
                r[i] = s
                return model(r, physics, **kwargs)

            loss += loss_fn(
                x=x, y=y[i], x_net=x_net, physics=physics[i], model=model_aux, **kwargs
            ).mean()

        return loss
