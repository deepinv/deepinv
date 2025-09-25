from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor
from deepinv.loss.adversarial.base import DiscriminatorMetric, AdversarialLoss
from deepinv.utils.mixins import MultiOperatorMixin

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics
    from deepinv.physics.generator.base import PhysicsGenerator


class UAIRLoss(MultiOperatorMixin, AdversarialLoss):
    r"""Reimplementation of UAIR generator's adversarial loss.

    The loss, introduced by :footcite:t:`pajot2019unsupervised`, is defined as follows, to be minimized by the generator:

    :math:`\mathcal{L}=\mathcal{L}_\text{adv}(\hat y, y;D)+\lambda\lVert \forw{\inverse{\hat y}}- \hat y\rVert^2_2,\quad\hat y=\forw{\hat x}`

    where :math:`\lambda` is a hyperparameter, and the standard adversarial loss is

    :math:`\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    where :math:`D` is the discriminator model and :math:`q` is the GAN metric between discriminator output and labels.

    In the multi-operator case, :math:`\forw{\cdot}` can be modified in the loss by passing a `physics_generator`.

    See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for examples of training generator and discriminator models.

    :param float weight_adv: weight for adversarial loss, defaults to 0.5 (from original paper)
    :param float weight_mc: weight for measurement consistency, defaults to 1.0 (from original paper)
    :param torch.nn.Module metric: metric for measurement consistency, defaults to :class:`torch.nn.MSELoss`
    :param str domain: if `None`, compute loss in measurement domain, if `A_adjoint` or `A_dagger`, map to image domain before computing loss.
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param deepinv.loss.adversarial.DiscriminatorMetric metric_gan: GAN metric :math:`q`. Defaults to
        :class:`deepinv.loss.adversarial.DiscriminatorMetric` which implements least squared metric as in LSGAN.
    :param torch.optim.Optimizer optimizer_D: optimizer for training discriminator.
        If `None` (default), do not train discriminator model.
    :param deepinv.physics.generator.PhysicsGenerator physics_generator: physics generator that returns new physics parameters
        If `None`, uses same physics every forward pass.

    .. warning::

        The physics generator cannot share the same random number generator as that of any previous physics generators,
        and the dataloader cannot be the same object as that of any previous dataloaders, otherwise
        this loss will affect data outside the loss.

    |sep|

    :Examples:

        Simple example (assuming a pretrained discriminator):

        >>> from deepinv.loss.adversarial import UAIRLoss
        >>> y, x_net = torch.randn(2, 1, 3, 64, 64) # B,C,H,W
        >>>
        >>> from deepinv.physics import Inpainting
        >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
        >>>
        >>> from deepinv.models import DCGANDiscriminator
        >>> D = DCGANDiscriminator() # assume pretrained discriminator
        >>>
        >>> # Assume physics is random masking
        >>> physics_generator = BernoulliSplittingMaskGenerator((64, 64), split_ratio=0.8)
        >>>
        >>> physics = Inpainting((2, 64, 64), mask=0.8)
        >>>
        >>> # Dataloader takes exact same form as input data
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader([(torch.randn(2, 64, 64), torch.randn(2, 64, 64)) for _ in range(2)]) # x, y
        >>>
        >>> loss = UAIRLoss(
        ...     D=D,
        ...     physics_generator=physics_generator,
        ... )
        >>>
        >>> from deepinv.models import MedianFilter
        >>> l = loss(y=y, x_net=x_net, physics=physics, model=MedianFilter())
        >>> l.backward()

    """

    def __init__(
        self,
        weight_adv: float = 0.5,
        weight_mc: float = 1,
        metric: Optional[nn.Module] = None,
        domain: str = None,
        D: nn.Module = None,
        metric_gan: DiscriminatorMetric = None,
        optimizer_D: torch.optim.Optimizer = None,
        physics_generator: PhysicsGenerator = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            weight_adv=weight_adv,
            device=device,
            metric_gan=metric_gan,
            optimizer_D=optimizer_D,
            **kwargs,
        )

        if metric is None:
            metric = nn.MSELoss()

        self.metric = metric
        self.domain = domain
        self.weight_mc = weight_mc
        self.D = D
        self.physics_generator = physics_generator

        if domain is not None and domain not in ("A_adjoint", "A_dagger"):
            raise ValueError("domain must be either None, A_adjoint or A_dagger.")

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics: Physics,
        model: nn.Module,
        *args,
        **kwargs,
    ):
        r"""Forward pass for UAIR generator's adversarial loss.

        :param torch.Tensor y: input measurement
        :param torch.Tensor y_hat: re-measured reconstruction
        :param deepinv.physics.Physics physics: forward physics
        :param torch.nn.Module model: reconstruction network
        """
        physics_new = self.next_physics(
            physics, physics_generator=self.physics_generator, batch_size=len(y)
        )
        y_hat = physics_new(x_net)
        x_tilde = model(y_hat, physics_new)
        y_tilde = physics_new.A(x_tilde)  # use same operator as y_hat

        if self.domain is not None:
            y = getattr(physics, self.domain)(y)
            y_hat = getattr(physics_new, self.domain)(y_hat)
            y_tilde = getattr(physics_new, self.domain)(y_tilde)

        with self.step_discrim(model) as step:
            step(self.adversarial_discrim(y, y_hat))

        adv_loss = self.adversarial_gen(y, y_hat)
        mc_loss = self.metric(y_tilde, y_hat)

        return adv_loss + mc_loss * self.weight_mc
