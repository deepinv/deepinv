import torch.nn as nn
from torch import Tensor
from deepinv.loss.loss import Loss
from .base import GeneratorLoss, DiscriminatorLoss
from deepinv.physics import Physics


class AmbientGANGeneratorLoss(GeneratorLoss):
    """Reimplementation of AmbientGAN generator's adversarial loss. See :class:`deepinv.loss.adversarial.GeneratorLoss` for how adversarial loss is calculated.

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


class AmbientGANDiscriminatorLoss(DiscriminatorLoss):
    """Reimplementation of AmbientGAN discriminator's adversarial loss. See :class:`deepinv.loss.adversarial.DiscriminatorLoss` for how adversarial loss is calculated.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 1.0, device="cpu"):
        super().__init__(weight_adv=weight_adv, device=device)
        self.name = "AmbientGANDiscriminator"

    def forward(self, y: Tensor, y_hat: Tensor, D: nn.Module, **kwargs):
        return self.adversarial_loss(y, y_hat, D)
