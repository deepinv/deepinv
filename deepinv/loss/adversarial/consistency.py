import torch.nn as nn
from torch import Tensor

from .base import GeneratorLoss, DiscriminatorLoss


class SupAdversarialGeneratorLoss(GeneratorLoss):
    """Supervised adversarial consistency loss for generator, as used in conditional GANs such as Kupyn et al., "DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks", and generative models such as Bora et al., "Compressed Sensing using Generative Models".

    Constructs adversarial loss between reconstructed image and the ground truth, to be minimised by generator.

    See ``deepinv.examples.adversarial_learning`` for formulae.

    :param float weight_adv: weight for adversarial loss, defaults to 0.01 (from original paper)
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 0.01, device="cpu", **kwargs):
        super().__init__(weight_adv=weight_adv, device=device, **kwargs)
        self.name = "SupAdversarialGenerator"

    def forward(self, x: Tensor, x_net: Tensor, D: nn.Module, **kwargs) -> Tensor:
        return self.adversarial_loss(x, x_net, D)


class SupAdversarialDiscriminatorLoss(DiscriminatorLoss):
    """Supervised adversarial consistency loss for discriminator, as used in conditional GANs such as Kupyn et al., "DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks", and generative models such as Bora et al., "Compressed Sensing using Generative Models".

    Constructs adversarial loss between reconstructed image and the ground truth, to be maximised by discriminator.

    See ``deepinv.examples.adversarial_learning`` for formulae.

    :param float weight: weight for adversarial loss, defaults to 1.0
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 1.0, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.name = "SupAdversarialDiscriminator"

    def forward(self, x: Tensor, x_net: Tensor, D: nn.Module, **kwargs) -> Tensor:
        return self.adversarial_loss(x, x_net, D) * 0.5


class UnsupAdversarialGeneratorLoss(GeneratorLoss):
    """Unsupervised adversarial consistency loss for generator, as used in unsupervised generative models such as Bora et al., "AmbientGAN: Generative models from lossy measurements".

    Constructs adversarial loss between input measurement and re-measured reconstruction, to be minimised by generator.

    See ``deepinv.examples.adversarial_learning`` for formulae.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(
        self,
        weight_adv: float = 1.0,
        device="cpu",
    ):
        super().__init__(weight_adv=weight_adv, device=device)
        self.name = "AmbientGANGenerator"

    def forward(self, y: Tensor, y_hat: Tensor, D: nn.Module, **kwargs):
        return self.adversarial_loss(y, y_hat, D)


class UnsupAdversarialDiscriminatorLoss(DiscriminatorLoss):
    """Unsupervised adversarial consistency loss for discriminator, as used in unsupervised generative models such as Bora et al., "AmbientGAN: Generative models from lossy measurements".

    Constructs adversarial loss between input measurement and re-measured reconstruction, to be maximised by discriminator.

    See ``deepinv.examples.adversarial_learning`` for formulae.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 1.0, device="cpu"):
        super().__init__(weight_adv=weight_adv, device=device)
        self.name = "AmbientGANDiscriminator"

    def forward(self, y: Tensor, y_hat: Tensor, D: nn.Module, **kwargs):
        return self.adversarial_loss(y, y_hat, D)
