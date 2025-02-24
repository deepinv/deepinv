from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from copy import deepcopy

from torch.utils.data import DataLoader
import torch.nn as nn
from torch import Tensor
from deepinv.loss.adversarial.base import GeneratorLoss, DiscriminatorLoss

if TYPE_CHECKING:
    from deepinv.physics.generator.base import PhysicsGenerator
    from deepinv.physics.forward import Physics


class MultiOperatorMixin:
    """Mixin for multi-operator loss functions.

    Pass in factory args for a physics generator or a dataloader to return new physics params or new data samples.

    :param callable physics_generator_factory: callable that returns a physics generator that returns new physics parameters
    :param callable dataloader_factory: callable that returns a dataloader that returns new samples
    """

    def __init__(
        self,
        physics_generator_factory: Callable[..., PhysicsGenerator] = None,
        dataloader_factory: Callable[..., DataLoader] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.physics_generator = None
        self.dataloader = None

        if physics_generator_factory is not None:
            self.physics_generator = physics_generator_factory()

        if dataloader_factory is not None:
            self.dataloader = dataloader_factory()
            self.prev_epoch = -1
            self.reset_iter(epoch=0)

    def next_physics(self, physics: Physics, batch_size=1) -> Physics:
        """Return physics with new physics params.

        :param deepinv.physics.Physics physics: old physics.
        :param int batch_size: batch size, defaults to 1
        :return deepinv.physics.Physics: new physics.
        """
        if self.physics_generator is not None:
            physics_cur = deepcopy(physics)
            params = self.physics_generator.step(batch_size=batch_size)
            physics_cur.update_parameters(**params)
            return physics_cur
        return physics

    def next_data(self) -> Tensor:
        """Return new data samples.
        :return torch.Tensor: new data samples.
        """
        if self.dataloader is not None:
            return next(self.iterator)

    def reset_iter(self, epoch: int) -> None:
        """Reset data iterator every epoch (to prevent `StopIteration`).
        :param int epoch: Epoch.
        """
        if epoch == self.prev_epoch + 1:
            self.iterator = iter(self.dataloader)
            self.prev_epoch += 1


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

    :param callable physics_generator_factory: callable that returns a physics generator that returns new physics parameters.
        If `None`, uses same physics every forward pass.
    :param float weight_adv: weight for adversarial loss, defaults to 0.5 (from original paper)
    :param float weight_mc: weight for measurement consistency, defaults to 1.0 (from original paper)
    :param torch.nn.Module metric: metric for measurement consistency, defaults to :class:`torch.nn.MSELoss`
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(
        self,
        weight_adv: float = 0.5,
        weight_mc: float = 1,
        metric: nn.Module = nn.MSELoss(),
        D: nn.Module = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(weight_adv=weight_adv, device=device, **kwargs)
        self.name = "UAIRGenerator"
        self.metric = metric
        self.weight_mc = weight_mc
        self.D = D

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

        adv_loss = self.adversarial_loss(y, y_hat, D)

        x_tilde = model(y_hat, physics_new)
        y_tilde = physics_new.A(x_tilde)  # use same operator as y_hat
        mc_loss = self.metric(y_tilde, y_hat)

        return adv_loss + mc_loss * self.weight_mc


class UAIRDiscriminatorLoss(MultiOperatorMixin, DiscriminatorLoss):
    """UAIR Discriminator's adversarial loss.

    For details and parameters, see :class:`deepinv.loss.adversarial.UAIRGeneratorLoss`
    """

    def __init__(
        self, weight_adv: float = 1.0, D: nn.Module = None, device="cpu", **kwargs
    ):
        super().__init__(weight_adv=weight_adv, D=D, device=device, **kwargs)
        self.name = "UAIRDiscriminator"

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics: Physics,
        model: nn.Module,
        D: nn.Module = None,
        **kwargs,
    ):
        y_hat = self.next_physics(physics, batch_size=len(y)).A(x_net)
        return self.adversarial_loss(y, y_hat, D)
