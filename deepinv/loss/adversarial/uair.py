import torch.nn as nn
from torch import Tensor
from .base import GeneratorLoss, DiscriminatorLoss
from deepinv.physics import Physics


class UAIRGeneratorLoss(GeneratorLoss):
    r"""Reimplementation of UAIR generator's adversarial loss.

    Pajot et al., "Unsupervised Adversarial Image Reconstruction".

    The loss is defined as follows, to be minimised by the generator:

    :math:`\mathcal{L}=\mathcal{L}_\text{adv}(\hat y, y;D)+\lVert A(f(\hat y))- \hat y\rVert^2_2,\quad\hat y=A(\hat x)`

    where the standard adversarial loss is

    :math:`\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]`

    See ``deepinv.examples.adversarial_learning`` for examples.

    :param float weight_adv: weight for adversarial loss, defaults to 0.5 (from original paper)
    :param float weight_mc: weight for measurement consistency, defaults to 1.0 (from original paper)
    :param nn.Module metric: metric for measurement consistency, defaults to nn.MSELoss
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(
        self,
        weight_adv: float = 0.5,
        weight_mc: float = 1,
        metric: nn.Module = nn.MSELoss(),
        device="cpu",
    ):
        super().__init__(weight_adv=weight_adv, device=device)
        self.name = "UAIRGenerator"
        self.metric = metric
        self.weight_mc = weight_mc

    def forward(
        self,
        y: Tensor,
        y_hat: Tensor,
        physics: Physics,
        model: nn.Module,
        D: nn.Module,
        **kwargs,
    ):
        adv_loss = self.adversarial_loss(y, y_hat, D)

        x_tilde = model(y_hat)
        y_tilde = physics.A(x_tilde)  # use same operator as y_hat
        mc_loss = self.metric(y_tilde, y_hat)

        return adv_loss + mc_loss * self.weight_mc
