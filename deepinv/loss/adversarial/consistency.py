from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from torch import Tensor

from deepinv.loss.adversarial.base import GeneratorLoss, DiscriminatorLoss
from deepinv.loss.adversarial.mo import MultiOperatorMixin
from deepinv.physics.mri import MRI

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics


class SupAdversarialGeneratorLoss(GeneratorLoss):
    r"""Supervised adversarial consistency loss for generator.

    This loss was used in conditional GANs such as Kupyn et al., "DeblurGAN: Blind Motion Deblurring Using
    Conditional Adversarial Networks", and generative models such as Bora et al., "Compressed Sensing using Generative
    Models".

    Constructs adversarial loss between reconstructed image and the ground truth, to be minimised by generator.

    :math:`\mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]`

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for examples of training generator and discriminator models.

    Simple example (assuming a pretrained discriminator):

    ::

        from deepinv.models import DCGANDiscriminator
        D = DCGANDiscriminator() # assume pretrained discriminator

        loss = SupAdversarialGeneratorLoss(D=D)

        l = loss(x, x_net)

        l.backward()

    :param float weight_adv: weight for adversarial loss, defaults to 0.01 (from original paper)
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(
        self, weight_adv: float = 0.01, D: nn.Module = None, device="cpu", **kwargs
    ):
        super().__init__(weight_adv=weight_adv, D=D, device=device, **kwargs)
        self.name = "SupAdversarialGenerator"

    def forward(self, x: Tensor, x_net: Tensor, D: nn.Module = None, **kwargs):
        r"""Forward pass for supervised adversarial generator loss.

        :param torch.Tensor x: ground truth image
        :param torch.Tensor x_net: reconstructed image
        :param torch.nn.Module D: discriminator model. If None, then D passed from __init__ used. Defaults to None.
        """
        return self.adversarial_loss(x, x_net, D)


class SupAdversarialDiscriminatorLoss(DiscriminatorLoss):
    r"""Supervised adversarial consistency loss for discriminator.

     This loss was as used in conditional GANs such as Kupyn et al., "DeblurGAN: Blind Motion Deblurring Using
     Conditional Adversarial Networks", and generative models such as Bora et al., "Compressed Sensing using Generative
     Models".

    Constructs adversarial loss between reconstructed image and the ground truth, to be maximised by discriminator.

    :math:`\mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]`

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for examples of training generator and discriminator models.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(
        self, weight_adv: float = 1.0, D: nn.Module = None, device="cpu", **kwargs
    ):
        super().__init__(weight_adv=weight_adv, D=D, device=device, **kwargs)
        self.name = "SupAdversarialDiscriminator"

    def forward(self, x: Tensor, x_net: Tensor, D: nn.Module = None, **kwargs):
        r"""Forward pass for supervised adversarial discriminator loss.

        :param torch.Tensor x: ground truth image
        :param torch.Tensor x_net: reconstructed image
        :param torch.nn.Module D: discriminator model. If None, then D passed from __init__ used. Defaults to None.
        """
        return self.adversarial_loss(x, x_net, D) * 0.5


class UnsupAdversarialGeneratorLoss(GeneratorLoss):
    r"""Unsupervised adversarial consistency loss for generator.

    This loss was used in unsupervised generative models such as Bora et al.,
    "AmbientGAN: Generative models from lossy measurements".

    Constructs adversarial loss between input measurement and re-measured reconstruction :math:`\hat{y}`, to be minimised by generator.

    :math:`\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for examples of training generator and discriminator models.

    We also provide the option to perform the loss calculation in the image domain using
    :math:`q(\cdot):=q(A^\top(\cdot))` or :math:`q(A^\dagger(\cdot))`. Note this requires your physics to be linear, e.g. MRI.

    Simple example (assuming a pretrained discriminator):

    ::

        from deepinv.models import DCGANDiscriminator
        D = DCGANDiscriminator() # assume pretrained discriminator

        loss = UnsupAdversarialGeneratorLoss(D=D)

        l = loss(y, y_hat)

        l.backward()

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param str metric: if `None`, compute loss in measurement domain, if `A_adjoint` or `A_dagger`, map to image domain before computing loss.
    """

    def __init__(
        self,
        weight_adv: float = 1.0,
        D: nn.Module = None,
        metric: str = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(weight_adv=weight_adv, D=D, device=device)
        self.name = "UnsupAdversarialGenerator"
        self.metric = metric
        if metric is not None and metric not in ("A_adjoint", "A_dagger"):
            raise ValueError("metric must be either None, A_adjoint or A_dagger.")

    def forward(
        self,
        y: Tensor,
        y_hat: Tensor,
        D: nn.Module = None,
        physics: Physics = None,
        **kwargs,
    ):
        r"""Forward pass for unsupervised adversarial generator loss.

        :param torch.Tensor y: input measurement
        :param torch.Tensor y_hat: re-measured reconstruction
        :param torch.Tensor x_net: reconstructed image
        :param torch.nn.Module D: discriminator model. If None, then D passed from __init__ used. Defaults to None.
        :param deepinv.physics.Physics physics: measurement operator.
        """
        if self.metric is not None:
            x_tilde = physics.__getattr__(self.metric)(y)
            x_hat = physics.__getattr__(self.metric)(y_hat)
            return self.adversarial_loss(x_tilde, x_hat, D)
        else:
            return self.adversarial_loss(y, y_hat, D)


class UnsupAdversarialDiscriminatorLoss(DiscriminatorLoss):
    r"""Unsupervised adversarial consistency loss for discriminator.

    This loss was used in unsupervised generative models such as
    Bora et al., "AmbientGAN: Generative models from lossy measurements".

    Constructs adversarial loss between input measurement and re-measured reconstruction, to be maximised
    by discriminator.

    :math:`\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for examples of training generator and discriminator models.

    We also provide the option to perform the loss calculation in the image domain using
    :math:`q(\cdot):=q(A^\top(\cdot))` or :math:`q(A^\dagger(\cdot))`. Note this requires your physics to be linear, e.g. MRI.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param str metric: if `None`, compute loss in measurement domain, if `A_adjoint` or `A_dagger`, map to image domain before computing loss.
    """

    def __init__(
        self,
        weight_adv: float = 1.0,
        D: nn.Module = None,
        metric: str = None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(weight_adv=weight_adv, D=D, device=device)
        self.name = "UnsupAdversarialDiscriminator"
        self.metric = metric
        if metric is not None and metric not in ("A_adjoint", "A_dagger"):
            raise ValueError("metric must be either None, A_adjoint or A_dagger.")

    def forward(
        self,
        y: Tensor,
        y_hat: Tensor,
        D: nn.Module = None,
        physics: Physics = None,
        **kwargs,
    ):
        r"""Forward pass for unsupervised adversarial discriminator loss.

        :param torch.Tensor y: input measurement
        :param torch.Tensor y_hat: re-measured reconstruction
        :param torch.Tensor x_net: reconstructed image
        :param torch.nn.Module D: discriminator model. If None, then D passed from __init__ used. Defaults to None.
        :param deepinv.physics.Physics physics: measurement operator.
        """
        if self.metric is not None:
            x_tilde = physics.__getattr__(self.metric)(y)
            x_hat = physics.__getattr__(self.metric)(y_hat)
            return self.adversarial_loss(x_tilde, x_hat, D)
        else:
            return self.adversarial_loss(y, y_hat, D)


class MultiOperatorUnsupAdversarialGeneratorLoss(
    MultiOperatorMixin, UnsupAdversarialGeneratorLoss
):
    """Multi-operator unsupervised adversarial loss for generator.

    Extends unsupervised adversarial loss by sampling new physics ("multi-operator") and new data every iteration.

    Proposed in `Fast Unsupervised MRI Reconstruction Without Fully-Sampled Ground Truth Data Using Generative Adversarial Networks <https://openaccess.thecvf.com/content/ICCV2021W/LCI/html/Cole_Fast_Unsupervised_MRI_Reconstruction_Without_Fully-Sampled_Ground_Truth_Data_Using_ICCVW_2021_paper.html>`_.
    The loss is constructed as follows, to be minimised by generator:

    :math:`\mathcal{L}_\text{adv}(\tilde{y},\hat y;D)=\mathbb{E}_{\tilde{y}\sim p_{\tilde{y}}}\left[q(D(\tilde{y}))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    where :math:`\hat y=A_2\hat x` is the re-measured reconstruction via a random operator :math:`A_2\sim\mathcal{A}`,
    and :math:`\tilde y` is a random measurement drawn from a dataset of measurements.

    We also provide the option to perform the loss calculation in the image domain using
    :math:`q(\cdot):=q(A^\top(\cdot))` or :math:`q(A^\dagger(\cdot))`. Note this requires your physics to be linear, e.g. MRI.

    Simple example (assuming a pretrained discriminator):

    ::

        from deepinv.models import SkipConvDiscriminator
        from deepinv.physics.generator import GaussianMaskGenerator
        from torch.utils.data import DataLoader

        D = SkipConvDiscriminator() # assume pretrained discriminator

        # Assume physics is random masking
        physics_generator_factory = lambda: GaussianMaskGenerator((64, 64))

        # Dataloader takes exact same form as input data
        dataloader_factory = lambda: DataLoader(...)

        # Generator loss
        loss = MultiOperatorUnsupAdversarialGeneratorLoss(
            D=D,
            physics_generator_factory=physics_generator_factory
            dataloader_factory=dataloader_factory
        )

        # Discriminator loss constructed in exactly same way
        loss_d = MultiOperatorUnsupAdversarialDiscriminatorLoss(
            D=D,
            physics_generator_factory=physics_generator_factory
            dataloader_factory=dataloader_factory
        )

        l = loss(y, y_hat, physics)
        l.backward(retain_graph=True)

        l_d = loss_d(y, y_hat, physics)
        l_d.backward()

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param Callable physics_generator_factory: callable that returns a physics generator that returns new physics parameters
    :param Callable dataloader_factory: callable that returns a dataloader that returns new samples
    :param str metric: if `None`, compute loss in measurement domain, if `A_adjoint` or `A_dagger`, map to image domain before computing loss.
    """

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics: Physics,
        D: nn.Module = None,
        epoch=None,
        **kwargs,
    ):
        self.reset_iter(epoch=epoch)

        y_tilde = self.next_data()[1].to(x_net.device)
        physics_new = self.next_physics(physics, batch_size=len(x_net))
        y_hat = physics_new.A(x_net)

        if y_tilde.shape != y_hat.shape:
            raise ValueError("Randomly sampled y_tilde must be same shape as y_hat.")

        if hasattr(physics, "mask") and torch.all(physics.mask == physics_new.mask):
            raise ValueError(
                "Randomly sampled physics should have different mask from orignal physics."
            )

        if self.metric is not None:
            physics_full = self.physics_like(physics)
            x_tilde = physics_full.__getattr__(self.metric)(y_tilde)
            x_hat = physics_new.__getattr__(self.metric)(y_hat)
            return self.adversarial_loss(x_tilde, x_hat, D)
        else:
            return self.adversarial_loss(y_tilde, y_hat, D)


class MultiOperatorUnsupAdversarialDiscriminatorLoss(
    MultiOperatorMixin, UnsupAdversarialDiscriminatorLoss
):
    """Multi-operator unsupervised adversarial loss for discriminator.

    Extends unsupervised adversarial loss by sampling new physics ("multi-operator") and new data every iteration.

    Proposed in `Fast Unsupervised MRI Reconstruction Without Fully-Sampled Ground Truth Data Using Generative Adversarial Networks <https://openaccess.thecvf.com/content/ICCV2021W/LCI/html/Cole_Fast_Unsupervised_MRI_Reconstruction_Without_Fully-Sampled_Ground_Truth_Data_Using_ICCVW_2021_paper.html>`_.
    The loss is constructed as follows, to be maximised by discriminator:

    :math:`\mathcal{L}_\text{adv}(\tilde{y},\hat y;D)=\mathbb{E}_{\tilde{y}\sim p_{\tilde{y}}}\left[q(D(\tilde{y}))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    where :math:`\hat y=A_2\hat x` is the re-measured reconstruction via a random operator :math:`A_2\sim\mathcal{A}`,
    and :math:`\tilde y` is a random measurement drawn from a dataset of measurements.

    We also provide the option to perform the loss calculation in the image domain using
    :math:`q(\cdot):=q(A^\top(\cdot))` or :math:`q(A^\dagger(\cdot))`. Note this requires your physics to be linear, e.g. MRI.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param Callable physics_generator_factory: callable that returns a physics generator that returns new physics parameters
    :param Callable dataloader_factory: callable that returns a dataloader that returns new samples
    :param str metric: if `None`, compute loss in measurement domain, if `A_adjoint` or `A_dagger`, map to image domain before computing loss.
    """

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics: Physics,
        D: nn.Module = None,
        epoch=None,
        **kwargs,
    ):
        self.reset_iter(epoch=epoch)

        y_tilde = self.next_data()[1].to(x_net.device)
        physics_new = self.next_physics(physics, batch_size=len(x_net))
        y_hat = physics_new.A(x_net)

        if self.metric is not None:
            physics_full = self.physics_like(physics)
            x_tilde = physics_full.__getattr__(self.metric)(y_tilde)
            x_hat = physics_new.__getattr__(self.metric)(y_hat)
            return self.adversarial_loss(x_tilde, x_hat, D)
        else:
            return self.adversarial_loss(y_tilde, y_hat, D)
