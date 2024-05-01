from contextlib import nullcontext

import torch.nn as nn
import torch
from torch import Tensor
from deepinv.loss.loss import Loss


class DiscriminatorMetric:
    r"""Generic GAN discriminator metric building block.

    Compares discriminator output with labels depending on if the image should be real or not.

    :param nn.Module metric: loss with which to compare outputs, defaults to nn.MSELoss()
    :param float real_label: value for ideal real image, defaults to 1.
    :param float fake_label: value for ideal fake image, defaults to 0.
    :param bool no_grad: whether to no_grad the metric computation, defaults to False
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(
        self,
        metric: nn.Module = nn.MSELoss(),
        real_label: float = 1.0,
        fake_label: float = 0.0,
        no_grad: bool = False,
        device="cpu",
    ):
        self.real = Tensor([real_label]).to(device)
        self.fake = Tensor([fake_label]).to(device)
        self.no_grad = no_grad
        self.metric = metric

    def __call__(self, pred: Tensor, real: bool = None) -> Tensor:
        r"""Call discriminator loss.

        :param torch.Tensor pred: discriminator classification output
        :param bool real: whether image should be real or not, defaults to None
        :return torch.Tensor: loss value
        """
        target = (self.real if real else self.fake).expand_as(pred)
        with torch.no_grad() if self.no_grad else nullcontext():
            return self.metric(pred, target)


class GeneratorLoss(Loss):
    r"""Base generator adversarial loss. Override the forward function to
    call `adversarial_loss` with quantities depending on your specific GAN model.
    For examples, see :class:`deepinv.loss.adversarial.SupAdversarialGeneratorLoss`, :class:`deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss`

    See ``deepinv.examples.adversarial_learning`` for formulae.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 1.0, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.metric_gan = DiscriminatorMetric(device=device)
        self.weight_adv = weight_adv

    def adversarial_loss(self, real: Tensor, fake: Tensor, D: nn.Module) -> Tensor:
        r"""Typical adversarial loss in GAN generators.

        :param Tensor real: image labelled as real, typically one originating from training set
        :param Tensor fake: image labelled as fake, typically a reconstructed image
        :param nn.Module D: discriminator/critic/classifier model
        :return Tensor: generator adversarial loss
        """
        pred_fake = D(fake)
        return self.metric_gan(pred_fake, real=True) * self.weight_adv

    def forward(self, *args, **kwargs) -> Tensor:
        return NotImplementedError()


class DiscriminatorLoss(Loss):
    r"""Base discriminator adversarial loss. Override the forward function to
    call `adversarial_loss` with quantities depending on your specific GAN model.
    For examples, see :class:`deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss`, :class:`deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss`.

    See ``deepinv.examples.adversarial_learning`` for formulae.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(self, weight_adv: float = 1.0, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.metric_gan = DiscriminatorMetric(device=device)
        self.weight_adv = weight_adv

    def adversarial_loss(self, real: Tensor, fake: Tensor, D: nn.Module):
        r"""Typical adversarial loss in GAN discriminators.

        :param Tensor real: image labelled as real, typically one originating from training set
        :param Tensor fake: image labelled as fake, typically a reconstructed image
        :param nn.Module D: discriminator/critic/classifier model
        :return Tensor: discriminator adversarial loss
        """
        pred_real = D(real)
        pred_fake = D(fake.detach())

        adv_loss_real = self.metric_gan(pred_real, real=True)
        adv_loss_fake = self.metric_gan(pred_fake, real=False)

        return (adv_loss_real + adv_loss_fake) * self.weight_adv

    def forward(self, *args, **kwargs) -> Tensor:
        return NotImplementedError()