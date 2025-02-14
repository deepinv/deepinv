from contextlib import nullcontext

import torch.nn as nn
import torch
from torch import Tensor
from deepinv.loss.loss import Loss


class DiscriminatorMetric:
    r"""
    Generic GAN discriminator metric building block.

    Compares discriminator output with labels depending on if the image should be real or not.

    The loss function is composed following LSGAN:
    `Least Squares Generative Adversarial Networks <https://arxiv.org/abs/1611.04076v3>`_

    This can be overriden to provide any flavour of discriminator metric, e.g. NSGAN, WGAN, LSGAN etc.

    See `Are GANs Created Equal? <https://arxiv.org/abs/1711.10337>`_ for a comparison.

    :param torch.nn.Module metric: loss with which to compare outputs, defaults to :class:`torch.nn.MSELoss`
    :param float real_label: value for ideal real image, defaults to 1.
    :param float fake_label: value for ideal fake image, defaults to 0.
    :param bool no_grad: whether to no_grad the metric computation, defaults to ``False``
    :param str device: torch device, defaults to ``"cpu"``
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
        r"""
        Call discriminator loss.

        :param torch.Tensor pred: discriminator classification output
        :param bool real: whether image should be real or not, defaults to `None`
        :return: loss value
        """
        target = (self.real if real else self.fake).expand_as(pred)
        with torch.no_grad() if self.no_grad else nullcontext():
            return self.metric(pred, target)


class GeneratorLoss(Loss):
    r"""Base generator adversarial loss.

    Override the forward function to call :func:`adversarial_loss <deepinv.loss.adversarial.GeneratorLoss.adversarial_loss>`
    with quantities depending on your specific GAN model.
    For examples, see :class:`SupAdversarialGeneratorLoss <deepinv.loss.adversarial.SupAdversarialGeneratorLoss>`
    and :class:`UnsupAdversarialGeneratorLoss <deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss>`

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for formulae.

    :param float weight_adv: weight for adversarial loss, defaults to 1.
    :param torch.nn.Module D: discriminator network. If not specified, `D` must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to `"cpu"`
    """

    def __init__(
        self, weight_adv: float = 1.0, D: nn.Module = None, device="cpu", **kwargs
    ):
        super().__init__(**kwargs)
        self.metric_gan = DiscriminatorMetric(device=device)
        self.weight_adv = weight_adv
        self.D = D

    def adversarial_loss(
        self, real: Tensor, fake: Tensor, D: nn.Module = None
    ) -> torch.Tensor:
        r"""Typical adversarial loss in GAN generators.

        :param torch.Tensor real: image labelled as real, typically one originating from training set
        :param torch.Tensor fake: image labelled as fake, typically a reconstructed image
        :param torch.nn.Module D: discriminator/critic/classifier model. If `None`, then `D` passed from `__init__` used.
            Defaults to `None`.
        :return: generator adversarial loss
        """
        D = self.D if D is None else D

        pred_fake = D(fake)
        return self.metric_gan(pred_fake, real=True) * self.weight_adv

    def forward(self, *args, D: nn.Module = None, **kwargs) -> torch.Tensor:
        return NotImplementedError()


class DiscriminatorLoss(Loss):
    r"""
    Base discriminator adversarial loss.

    Override the forward function to
    call ``adversarial_loss`` with quantities depending on your specific GAN model.

    For examples, see :class:`deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss`,
    :class:`deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss`.

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for formulae.

    :param float weight_adv: weight for adversarial loss, defaults to 1.
    :param torch.nn.Module D: discriminator network.
        If not specified, ``D`` must be provided in ``forward``, defaults to ``None``.
    :param str device: torch device, defaults to `"cpu"`.
    """

    def __init__(
        self, weight_adv: float = 1.0, D: nn.Module = None, device="cpu", **kwargs
    ):
        super().__init__(**kwargs)
        self.metric_gan = DiscriminatorMetric(device=device)
        self.weight_adv = weight_adv
        self.D = D

    def adversarial_loss(self, real: Tensor, fake: Tensor, D: nn.Module = None):
        r"""Typical adversarial loss in GAN discriminators.

        :param torch.Tensor real: image labelled as real, typically one originating from training set
        :param torch.Tensor fake: image labelled as fake, typically a reconstructed image
        :param torch.nn.Module D: discriminator/critic/classifier model. If None, then D passed from __init__ used. Defaults to None.
        :return: (:class:`torch.Tensor`) discriminator adversarial loss
        """
        D = self.D if D is None else D

        pred_real = D(real)
        pred_fake = D(fake.detach())

        adv_loss_real = self.metric_gan(pred_real, real=True)
        adv_loss_fake = self.metric_gan(pred_fake, real=False)

        return (adv_loss_real + adv_loss_fake) * self.weight_adv

    def forward(self, *args, D: nn.Module = None, **kwargs) -> torch.Tensor:
        return NotImplementedError()
