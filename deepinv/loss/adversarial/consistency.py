from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from deepinv.loss.adversarial.base import DiscriminatorMetric, AdversarialLoss
from deepinv.utils.mixins import MultiOperatorMixin

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics
    from deepinv.physics.generator.base import PhysicsGenerator


class SupAdversarialLoss(AdversarialLoss):
    r"""Supervised adversarial consistency loss for generator.

    This loss was as used in conditional GANs such as :footcite:t:`kupyn2018deblurgan` and generative models such as :footcite:t:`bora2017compressed`.

    Constructs adversarial loss between reconstructed image and the ground truth, to be minimised by generator
    (and maximised by discriminator:

    :math:`\mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]`

    where :math:`D` is the discriminator model and :math:`q` is the GAN metric between discriminator output and labels.

    See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for examples of training generator and discriminator models.

    :param float weight_adv: weight for adversarial loss, defaults to 0.01 (from original paper)
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param deepinv.loss.adversarial.DiscriminatorMetric metric_gan: GAN metric :math:`q`. Defaults to
        :class:`deepinv.loss.adversarial.DiscriminatorMetric` which implements least squared metric as in LSGAN.
    :param torch.optim.Optimizer optimizer_D: optimizer for training discriminator.
        If `None` (default), do not train discriminator model.
    :param torch.optim.lr_scheduler.LRScheduler scheduler_D: optional learning rate scheduler
        for discriminator. If optimizer not passed, then this is ignored.
    :param str device: torch device, defaults to "cpu"

    |sep|

    :Examples:

        Simple example (assuming a pretrained discriminator):

        >>> x, x_net = torch.randn(2, 1, 3, 64, 64) # B,C,H,W
        >>>
        >>> from deepinv.models import DCGANDiscriminator
        >>> D = DCGANDiscriminator() # assume pretrained discriminator
        >>> from deepinv.loss.adversarial import SupAdversarialLoss
        >>> loss = SupAdversarialLoss(D=D)
        >>> l = loss(x=x, x_net=x_net)
        >>> l.backward()
    """

    def forward(
        self, x: Tensor, x_net: Tensor, model: nn.Module = None, *args, **kwargs
    ):
        r"""Forward pass for supervised adversarial generator loss.

        :param torch.Tensor x: ground truth image
        :param torch.Tensor x_net: reconstructed image
        :param torch.nn.Module model: reconstruction network
        """
        with self.step_discrim(model) as step:
            for _ in range(self.num_D_steps):
                step(self.adversarial_discrim(x, x_net))

        return self.adversarial_gen(x, x_net)


class GPLoss(SupAdversarialLoss):
    r"""
    WGAN-GP loss extending AdversarialLoss.

    Implements the gradient penalty term from:
    Gulrajani et al., "Improved Training of Wasserstein GANs" (2017).

    Critic loss:
        L_D = L_D_base + λ * E[(||∇_x̂ D(x̂)||_2 - 1)^2]

    Generator loss:
        L_G = -E[D(fake)]  (handled by base class)
    """

    def __init__(self, lambda_gp: float = 10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Compute the gradient penalty term."""
        batch_size = real.size(0)
        epsilon = torch.rand(
            batch_size, 1, 1, 1, device=self.device, requires_grad=True
        )
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)

        pred_interpolated = self.D(interpolated)
        gradients = torch.autograd.grad(
            outputs=pred_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(pred_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp

    def forward(
        self,
        x: torch.Tensor,
        x_net: torch.Tensor,
        model: nn.Module = None,
        *args,
        **kwargs,
    ):
        with self.step_discrim(model) as step:
            for _ in range(self.num_D_steps):
                l = self.adversarial_discrim(x, x_net)
                if model.training:
                    l += self.lambda_gp * self.gradient_penalty(x, x_net)
                step(l)

        return self.adversarial_gen(x, x_net)


class UnsupAdversarialLoss(AdversarialLoss):
    r"""Unsupervised adversarial consistency loss for generator.

    This loss was used for unsupervised generative models such as in :footcite:t:`bora2018ambientgan`.

    Constructs adversarial loss between input measurement and re-measured reconstruction :math:`\hat{y}`, to be minimised by generator,
    (and maximised by discriminator:

    :math:`\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    where :math:`D` is the discriminator model and :math:`q` is the GAN metric between discriminator output and labels.

    See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for examples of training generator and discriminator models.

    We also provide the option to perform the loss calculation in the image domain using
    :math:`q(\cdot):=q(A^\top(\cdot))` or :math:`q(A^\dagger(\cdot))`.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param str domain: if `None`, compute loss in measurement domain, if :func:`A_adjoint <deepinv.physics.LinearPhysics.A_adjoint>` or :func:`A_dagger <deepinv.physics.Physics.A_dagger>`, map to image domain before computing loss.
    :param deepinv.loss.adversarial.DiscriminatorMetric metric_gan: GAN metric :math:`q`. Defaults to
        :class:`deepinv.loss.adversarial.DiscriminatorMetric` which implements least squared metric as in LSGAN.
    :param torch.optim.Optimizer optimizer_D: optimizer for training discriminator.
        If `None` (default), do not train discriminator model.
    :param torch.optim.lr_scheduler.LRScheduler scheduler_D: optional learning rate scheduler
        for discriminator. If optimizer not passed, then this is ignored.

    |sep|

    :Examples:

        Simple example (assuming a pretrained discriminator):

        >>> from deepinv.models import DCGANDiscriminator
        >>> from deepinv.loss.adversarial import UnsupAdversarialLoss
        >>> from deepinv.physics import Denoising
        >>>
        >>> x, x_net = torch.randn(2, 1, 3, 64, 64) # B,C,H,W
        >>> physics = Denoising()
        >>> y = physics(x)
        >>>
        >>> D = DCGANDiscriminator() # assume pretrained discriminator
        >>> loss = UnsupAdversarialLoss(D=D)
        >>> l = loss(y=y, x_net=x_net, physics=physics)
        >>> l.backward()
    """

    def __init__(
        self,
        weight_adv: float = 1.0,
        D: nn.Module = None,
        domain: str = None,
        metric_gan: DiscriminatorMetric = None,
        optimizer_D: torch.optim.Optimizer = None,
        scheduler_D: torch.optim.lr_scheduler.LRScheduler = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            weight_adv=weight_adv,
            D=D,
            metric_gan=metric_gan,
            device=device,
            optimizer_D=optimizer_D,
            scheduler_D=scheduler_D,
        )
        self.domain = domain
        if domain is not None and domain not in ("A_adjoint", "A_dagger"):
            raise ValueError("domain must be either None, A_adjoint or A_dagger.")

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics: Physics = None,
        model: nn.Module = None,
        *args,
        **kwargs,
    ):
        r"""Forward pass for unsupervised adversarial generator loss.

        :param torch.Tensor y: input measurement
        :param torch.Tensor x_net: reconstructed image
        :param deepinv.physics.Physics physics: measurement operator.
        :param torch.nn.Module model: reconstruction network
        """
        y_hat = physics.A(x_net)

        if self.domain is not None:
            x_tilde = getattr(physics, self.domain)(y)
            x_hat = getattr(physics, self.domain)(y_hat)

            with self.step_discrim(model) as step:
                step(self.adversarial_discrim(x_tilde, x_hat))

            return self.adversarial_gen(x_tilde, x_hat)
        else:
            with self.step_discrim(model) as step:
                step(self.adversarial_discrim(y, y_hat))

            return self.adversarial_gen(y, y_hat)


class MultiOperatorUnsupAdversarialLoss(UnsupAdversarialLoss, MultiOperatorMixin):
    r"""Multi-operator unsupervised adversarial loss for generator.

    Extends unsupervised adversarial loss by sampling new physics ("multi-operator") and new data every iteration.

    Proposed in :footcite:t:`cole2021fast`.
    The loss is constructed as follows, to be minimised by generator (and maximised by discriminator:

    :math:`\mathcal{L}_\text{adv}(\tilde{y},\hat y;D)=\mathbb{E}_{\tilde{y}\sim p_{\tilde{y}}}\left[q(D(\tilde{y}))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    where :math:`D` is the discriminator model and :math:`q` is the GAN metric between discriminator output and labels,
    :math:`\hat y=A_2\hat x` is the re-measured reconstruction via a random operator :math:`A_2\sim\mathcal{A}`,
    and :math:`\tilde y` is a random measurement drawn from a dataset of measurements.

    We also provide the option to perform the loss calculation in the image domain using
    :math:`q(\cdot):=q(A^\top(\cdot))` or :math:`q(A^\dagger(\cdot))`.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param deepinv.physics.generator.PhysicsGenerator physics_generator: physics generator that returns new physics parameters
    :param torch.utils.data.DataLoader dataloader: dataloader that returns new samples
    :param str domain: if `None`, compute loss in measurement domain, if :func:`A_adjoint <deepinv.physics.LinearPhysics.A_adjoint>` or :func:`A_dagger <deepinv.physics.Physics.A_dagger>`, map to image domain before computing loss.
    :param deepinv.loss.adversarial.DiscriminatorMetric metric_gan: GAN metric :math:`q`. Defaults to
        :class:`deepinv.loss.adversarial.DiscriminatorMetric` which implements least squared metric as in LSGAN.
    :param torch.optim.Optimizer optimizer_D: optimizer for training discriminator.
        If `None` (default), do not train discriminator model.
    :param torch.optim.lr_scheduler.LRScheduler scheduler_D: optional learning rate scheduler
        for discriminator. If optimizer not passed, then this is ignored.
    :param deepinv.physics.generator.PhysicsGenerator physics_generator: physics generator that returns new physics parameters
    :param torch.utils.data.DataLoader dataloader: dataloader that returns new samples

    .. warning::

        The physics generator cannot share the same random number generator as that of any previous physics generators,
        and the dataloader cannot be the same object as that of any previous dataloaders, otherwise
        this loss will affect data outside the loss.

    |sep|

    :Examples:

        Simple example (assuming a pretrained discriminator):

        >>> y = torch.randn(1, 2, 64, 64) # B,C,H,W
        >>> x_net = torch.randn(1, 2, 64, 64) # B,C,H,W
        >>>
        >>> from deepinv.models import SkipConvDiscriminator
        >>> from deepinv.physics import MRI
        >>> from deepinv.physics.generator import GaussianMaskGenerator
        >>> from deepinv.models import MedianFilter # any model
        >>> from torch.utils.data import DataLoader
        >>>
        >>> D = SkipConvDiscriminator(img_size=(64, 64), in_channels=2) # assume pretrained discriminator
        >>>
        >>> # Assume physics is random masking
        >>> physics_generator = GaussianMaskGenerator((64, 64))
        >>>
        >>> physics = MRI(img_size=(2, 64, 64))
        >>>
        >>> # Dataloader takes exact same form as input data
        >>> dataloader = DataLoader([(torch.randn(2, 64, 64), torch.randn(2, 64, 64)) for _ in range(2)]) # x, y
        >>>
        >>> from deepinv.loss.adversarial import MultiOperatorUnsupAdversarialLoss
        >>> loss = MultiOperatorUnsupAdversarialLoss(
        ...     D=D,
        ...     physics_generator=physics_generator,
        ...     dataloader=dataloader
        ... )
        >>>
        >>> l = loss(y=y, x_net=x_net, physics=physics, model=MedianFilter())
        >>> l.backward()
    """

    def __init__(
        self,
        weight_adv: float = 1.0,
        D: nn.Module = None,
        domain: str = None,
        metric_gan: DiscriminatorMetric = None,
        optimizer_D: torch.optim.Optimizer = None,
        scheduler_D: torch.optim.lr_scheduler.LRScheduler = None,
        physics_generator: PhysicsGenerator = None,
        dataloader: DataLoader = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            weight_adv=weight_adv,
            D=D,
            domain=domain,
            metric_gan=metric_gan,
            optimizer_D=optimizer_D,
            scheduler_D=scheduler_D,
            device=device,
        )
        self.physics_generator = physics_generator
        self.dataloader = dataloader

        if dataloader is None:
            raise ValueError("Dataloader must not be None.")

        self.prev_epoch = -1
        self.reset_iter(epoch=0)

    def reset_iter(self, epoch: int) -> None:
        """Reset data iterator every epoch (to prevent `StopIteration`).
        :param int epoch: Epoch.
        """
        if epoch == self.prev_epoch + 1:
            self.iterator = iter(self.dataloader)
            self.prev_epoch += 1

    def forward(
        self,
        x_net: Tensor,
        physics: Physics,
        model: nn.Module,
        epoch=None,
        *args,
        **kwargs,
    ):
        self.reset_iter(epoch=epoch)

        # Step data and physics
        y_tilde = next(self.iterator)[1].to(x_net.device)
        physics_new = self.next_physics(
            physics, physics_generator=self.physics_generator, batch_size=len(x_net)
        )

        y_hat = physics_new.A(x_net)

        if y_tilde.shape != y_hat.shape:
            raise ValueError("Randomly sampled y_tilde must be same shape as y_hat.")

        if hasattr(physics, "mask") and torch.all(physics.mask == physics_new.mask):
            raise ValueError(
                "Randomly sampled physics should have different mask from orignal physics."
            )

        if self.domain is not None:
            # Convert to domain, setting any masks to ones
            physics_full = physics.clone()
            if hasattr(physics, "mask"):
                if isinstance(physics.mask, Tensor):
                    physics_new.update(mask=torch.ones_like(physics.mask))
                elif isinstance(physics.mask, float):
                    physics_new.update(mask=1.0)

            x_tilde = getattr(physics_full, self.domain)(y_tilde)
            x_hat = getattr(physics_new, self.domain)(y_hat)

            with self.step_discrim(model) as step:
                step(self.adversarial_discrim(x_tilde, x_hat))

            return self.adversarial_gen(x_tilde, x_hat)
        else:
            with self.step_discrim(model) as step:
                step(self.adversarial_discrim(y_tilde, y_hat))

            return self.adversarial_gen(y_tilde, y_hat)
