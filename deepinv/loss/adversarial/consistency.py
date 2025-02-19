import torch.nn as nn
from torch import Tensor

from .base import GeneratorLoss, DiscriminatorLoss


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
    """

    def __init__(
        self,
        weight_adv: float = 1.0,
        D: nn.Module = None,
        device="cpu",
    ):
        super().__init__(weight_adv=weight_adv, D=D, device=device)
        self.name = "UnsupAdversarialGenerator"

    def forward(self, y: Tensor, y_hat: Tensor, D: nn.Module = None, **kwargs):
        r"""Forward pass for unsupervised adversarial generator loss.

        :param torch.Tensor y: input measurement
        :param torch.Tensor y_hat: re-measured reconstruction
        :param torch.nn.Module D: discriminator model. If None, then D passed from __init__ used. Defaults to None.
        """
        return self.adversarial_loss(y, y_hat, D)


class UnsupAdversarialDiscriminatorLoss(DiscriminatorLoss):
    r"""Unsupervised adversarial consistency loss for discriminator.

    This loss was used in unsupervised generative models such as
    Bora et al., "AmbientGAN: Generative models from lossy measurements".

    Constructs adversarial loss between input measurement and re-measured reconstruction, to be maximised
    by discriminator.

    :math:`\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for examples of training generator and discriminator models.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 1.0, D: nn.Module = None, device="cpu"):
        super().__init__(weight_adv=weight_adv, D=D, device=device)
        self.name = "UnsupAdversarialDiscriminator"

    def forward(self, y: Tensor, y_hat: Tensor, D: nn.Module = None, **kwargs):
        r"""Forward pass for unsupervised adversarial discriminator loss.

        :param torch.Tensor y: input measurement
        :param torch.Tensor y_hat: re-measured reconstruction
        :param torch.nn.Module D: discriminator model. If None, then D passed from __init__ used. Defaults to None.
        """
        return self.adversarial_loss(y, y_hat, D)
