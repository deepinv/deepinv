import torch.nn as nn
from torch import Tensor
from deepinv.loss.loss import Loss
from .base import GeneratorLoss, DiscriminatorLoss
from deepinv.physics import Physics


class UAIRGeneratorLoss(GeneratorLoss):
    """Reimplementation of UAIR generator's adversarial loss. See :class:`deepinv.loss.adversarial.GeneratorLoss` for how adversarial loss is calculated.

    :param float weight_adv: weight for adversarial loss, defaults to 0.5 (from original paper)
    :param float weight_mc: weight for measurement consistency, defaults to 1.0 (from original paper)
    :param nn.Module metric: metric for measurement consistency, defaults to nn.MSELoss
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(
        self,
        weight_adv: float = 0.5,
        weight_mc: float = 1,
        metric: nn.Module = nn.MSELoss,
        device="cpu",
    ):
        super().__init__(weight_adv=weight_adv, device=device)
        self.name = "UAIRGenerator"
        self.metric = metric
        self.weight_mc = weight_mc

    def forward(
        self, y: Tensor, y_hat: Tensor, physics: Physics, model: nn.Module, D: nn.Module, **kwargs
    ):
        adv_loss = self.adversarial_loss(y, y_hat, D)

        x_tilde = model(y_hat)
        y_tilde = physics.A(x_tilde)  # use same operator as y_hat
        mc_loss = self.metric(y_tilde, y_hat)

        return adv_loss + mc_loss * self.weight_mc


class UAIRDiscriminatorLoss(DiscriminatorLoss):
    """Reimplementation of UAIR discriminator's adversarial loss. See :class:`deepinv.loss.adversarial.DiscriminatorLoss` for how adversarial loss is calculated.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 1.0, device="cpu"):
        super().__init__(weight_adv=weight_adv, device=device)
        self.name = "UAIRDiscriminator"

    def forward(self, y: Tensor, y_hat: Tensor, D: nn.Module, **kwargs):
        return self.adversarial_loss(y, y_hat, D)
