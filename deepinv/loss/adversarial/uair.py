from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from deepinv.loss.adversarial.base import (
    GeneratorLoss,
    DiscriminatorLoss,
    DiscriminatorMetric,
)
from deepinv.loss.adversarial.mo import MultiOperatorMixin

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics


class UAIRGeneratorLoss(MultiOperatorMixin, GeneratorLoss):
    r"""Reimplementation of UAIR generator's adversarial loss.

    Pajot et al., "Unsupervised Adversarial Image Reconstruction".

    The loss is defined as follows, to be minimised by the generator:

    :math:`\mathcal{L}=\mathcal{L}_\text{adv}(\hat y, y;D)+\lambda\lVert \forw{\inverse{\hat y}}- \hat y\rVert^2_2,\quad\hat y=\forw{\hat x}`

    where :math:`\lambda` is a hyperparameter, and the standard adversarial loss is

    :math:`\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    where :math:`D` is the discriminator model and :math:`q` is the GAN metric between discriminator output and labels.

    In the multi-operator case, :math:`\forw{\cdot}` can be modified in the loss by passing a `physics_generator`.

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for examples of training generator and discriminator models.

    |sep|

    :Examples:

        Simple example (assuming a pretrained discriminator):

        >>> y, x_net = torch.randn(2, 1, 3, 64, 64) # B,C,H,W
        >>>
        >>> from deepinv.models import DCGANDiscriminator
        >>> D = DCGANDiscriminator() # assume pretrained discriminator
        >>>
        >>> from deepinv.physics import Inpainting
        >>> physics = Inpainting((3, 64, 64))
        >>>
        >>> loss = UAIRGeneratorLoss(D=D)
        >>> model = lambda y, physics: physics.A_adjoint(y)
        >>> l = loss(y, x_net, physics, model)
        >>> l.backward()


    :param float weight_adv: weight for adversarial loss, defaults to 0.5 (from original paper)
    :param float weight_mc: weight for measurement consistency, defaults to 1.0 (from original paper)
    :param torch.nn.Module metric: metric for measurement consistency, defaults to :class:`torch.nn.MSELoss`
    :param str metric_adv: if `None`, compute loss in measurement domain, if `A_adjoint` or `A_dagger`, map to image domain before computing loss.
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param deepinv.loss.adversarial.DiscriminatorMetric metric_gan: GAN metric :math:`q`. Defaults to
        :class:`deepinv.loss.adversarial.DiscriminatorMetric` which implements least squared metric as in LSGAN.
    :param deepinv.physics.generator.PhysicsGenerator physics_generator: physics generator that returns new physics parameters
        If `None`, uses same physics every forward pass.

    .. warning::

        When using `physics_generator is not None`, and generator loss in parallel with discriminator loss, the physics generators cannot share the same random number generator,
        otherwise both losses will step the same random number generators, meaning that the
        data seen by each loss will be different. A simple solution uses factories:

        ::

            rng_factory = lambda: torch.Generator(seed)
            physics_generator_factory = lambda: PhysicsGenerator(..., rng=rng_factory())
            gen_loss = UAIRGeneratorLoss(
                physics_generator = physics_generator_factory(),
            )
            dis_loss = UAIRDiscriminatorLoss(
                physics_generator = physics_generator_factory(),
            )

    """

    def __init__(
        self,
        weight_adv: float = 0.5,
        weight_mc: float = 1,
        metric: nn.Module = nn.MSELoss(),
        metric_adv: str = None,
        D: nn.Module = None,
        metric_gan: DiscriminatorMetric = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            weight_adv=weight_adv, device=device, metric_gan=metric_gan, **kwargs
        )
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
        y_hat = physics_new(x_net)
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
        metric_gan: DiscriminatorMetric = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            weight_adv=weight_adv, D=D, metric_gan=metric_gan, device=device, **kwargs
        )
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
        y_hat = physics_new(x_net)

        if self.metric_adv is not None:
            y = getattr(physics, self.metric_adv)(y)
            y_hat = getattr(physics_new, self.metric_adv)(y_hat)

        return self.adversarial_loss(y, y_hat, D)
