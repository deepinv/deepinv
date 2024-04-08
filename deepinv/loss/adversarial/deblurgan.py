import torch.nn as nn
from torch import Tensor

from .base import GeneratorLoss, DiscriminatorLoss


class DeblurGANGeneratorLoss(GeneratorLoss):
    """Reimplementation of DeblurGAN generator's adversarial loss. See :class:`deepinv.loss.adversarial.GeneratorLoss` for how adversarial loss is calculated.

    :param float weight_adv: weight for adversarial loss, defaults to 0.01 (from original paper)
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 0.01, device="cpu", **kwargs):
        super().__init__(weight_adv=weight_adv, device=device, **kwargs)
        self.name = "DeblurGANGenerator"

    def forward(self, x: Tensor, x_net: Tensor, D: nn.Module, **kwargs) -> Tensor:
        return self.adversarial_loss(x, x_net, D)


class DeblurGANDiscriminatorLoss(DiscriminatorLoss):
    """Reimplementation of DeblurGAN discriminator's adversarial loss. See :class:`deepinv.loss.adversarial.DiscriminatorLoss` for how adversarial loss is calculated.

    :param float weight: weight for adversarial loss, defaults to 1.0
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 1.0, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.name = "DeblurGANDiscriminator"

    def forward(self, x: Tensor, x_net: Tensor, D: nn.Module, **kwargs) -> Tensor:
        return self.adversarial_loss(x, x_net, D) * 0.5
