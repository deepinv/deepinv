from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import torch.nn as nn
from torch import Tensor
from deepinv.loss.adversarial.base import GeneratorLoss, DiscriminatorLoss
from deepinv.loss.adversarial.mo import MultiOperatorMixin

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics


class UAIRGeneratorLoss(MultiOperatorMixin, GeneratorLoss):
    r"""Reimplementation of UAIR generator's adversarial loss.

    Pajot et al., "Unsupervised Adversarial Image Reconstruction".

    The loss is defined as follows, to be minimised by the generator:

    :math:`\mathcal{L}=\mathcal{L}_\text{adv}(\hat y, y;D)+\lVert \forw{\inverse{\hat y}}- \hat y\rVert^2_2,\quad\hat y=\forw{\hat x}`

    where the standard adversarial loss is

    :math:`\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for examples of training generator and discriminator models.

    Simple example (assuming a pretrained discriminator):

    ::

        from deepinv.models import DCGANDiscriminator
        D = DCGANDiscriminator() # assume pretrained discriminator

        loss = UAIRGeneratorLoss(D=D)

        l = loss(y, y_hat, physics, model)

        l.backward()

    :param Callable physics_generator_factory: callable that returns a physics generator that returns new physics parameters.
        If `None`, uses same physics every forward pass.
    :param float weight_adv: weight for adversarial loss, defaults to 0.5 (from original paper)
    :param float weight_mc: weight for measurement consistency, defaults to 1.0 (from original paper)
    :param torch.nn.Module metric: metric for measurement consistency, defaults to :class:`torch.nn.MSELoss`
    :param str metric_adv: if `None`, compute loss in measurement domain, if `A_adjoint` or `A_dagger`, map to image domain before computing loss.
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(
        self,
        weight_adv: float = 0.5,
        weight_mc: float = 1,
        metric: nn.Module = nn.MSELoss(),
        metric_adv: str = None,
        D: nn.Module = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(weight_adv=weight_adv, device=device, **kwargs)
        self.name = "UAIRGenerator"
        self.metric = metric
        self.metric_adv = metric_adv
        self.weight_mc = weight_mc
        self.D = D

        if metric_adv is not None and metric_adv not in ("A_adjoint", "A_dagger"):
            raise ValueError("metric_adv must be either None, A_adjoint or A_dagger.")

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics: Physics,
        model: nn.Module,
        D: nn.Module = None,
        **kwargs,
    ):
        r"""Forward pass for UAIR generator's adversarial loss.

        :param torch.Tensor y: input measurement
        :param torch.Tensor y_hat: re-measured reconstruction
        :param deepinv.physics.Physics physics: forward physics
        :param torch.nn.Module model: reconstruction network
        :param torch.nn.Module D: discriminator model. If None, then D passed from __init__ used. Defaults to None.
        """
        D = self.D if D is None else D

        physics_new = self.next_physics(physics, batch_size=len(y))
        y_hat = physics_new.A(x_net)
        x_tilde = model(y_hat, physics_new)
        y_tilde = physics_new.A(x_tilde)  # use same operator as y_hat

        if self.metric_adv is not None:
            y = getattr(physics, self.metric_adv)(y)
            y_hat = getattr(physics_new, self.metric_adv)(y_hat)
            y_tilde = getattr(physics_new, self.metric_adv)(y_tilde)

        adv_loss = self.adversarial_loss(y, y_hat, D)
        mc_loss = self.metric(y_tilde, y_hat)

        return adv_loss + mc_loss * self.weight_mc


class UAIRDiscriminatorLoss(MultiOperatorMixin, DiscriminatorLoss):
    """UAIR Discriminator's adversarial loss.

    For details and parameters, see :class:`deepinv.loss.adversarial.UAIRGeneratorLoss`
    """

    def __init__(
        self,
        weight_adv: float = 1.0,
        metric_adv: str = None,
        D: nn.Module = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(weight_adv=weight_adv, D=D, device=device, **kwargs)
        self.name = "UAIRDiscriminator"
        self.metric_adv = metric_adv

        if metric_adv is not None and metric_adv not in ("A_adjoint", "A_dagger"):
            raise ValueError("metric_adv must be either None, A_adjoint or A_dagger.")

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics: Physics,
        model: nn.Module,
        D: nn.Module = None,
        **kwargs,
    ):
        physics_new = self.next_physics(physics, batch_size=len(y))
        y_hat = physics_new.A(x_net)

        if self.metric_adv is not None:
            y = getattr(physics, self.metric_adv)(y)
            y_hat = getattr(physics_new, self.metric_adv)(y_hat)

        return self.adversarial_loss(y, y_hat, D)
