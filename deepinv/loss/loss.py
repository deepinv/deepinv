from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics


class Loss(torch.nn.Module):
    r"""
    Base class for all loss functions.

    Sets a template for the loss functions, whose forward method must follow the input parameters in
    :meth:`deepinv.loss.Loss.forward`.
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
    ) -> Tensor:
        r"""
        Computes the loss.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor x: Reference image.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.

        :return: (torch.Tensor) loss, the tensor size might be (1,) or (batch size,).
        """
        raise NotImplementedError(
            "The method 'forward' must be implemented in the subclass."
        )

    def adapt_model(self, model: Module, **kwargs) -> Module:
        r"""
        Some loss functions require the model forward call to be adapted before the forward pass.

        :param torch.nn.Module model: reconstruction model
        """
        return model


class StackedPhysicsLoss(Loss):
    def __init__(self, losses):
        super(StackedPhysicsLoss, self).__init__()
        self.losses = losses

    def forward(self, y, x_net, physics, model, **kwargs):
        loss = 0
        for i, loss_fn in enumerate(self.losses):
            def model_aux(s, p, **kwargs):
                r = y.clone()
                r[i] = s
                return model(r, physics, **kwargs)
            loss += loss_fn(y=y[i], x_net=x_net, physics=physics[i], model=model_aux, **kwargs).mean()

        return loss