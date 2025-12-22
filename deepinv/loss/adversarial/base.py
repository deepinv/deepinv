from __future__ import annotations
from contextlib import nullcontext, contextmanager
from typing import TYPE_CHECKING
from pathlib import Path

import torch.nn as nn
import torch
from torch import Tensor
from deepinv.loss.loss import Loss
from deepinv.utils import AverageMeter

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics


class DiscriminatorMetric:
    r"""
    Generic GAN discriminator metric building block.

    Compares discriminator output with labels depending on if the image should be real or not.

    By default, the `metric` used is the MSE which gives the LSGAN from :footcite:t:`mao2017least`.

    Pass in a different `metric` or override `DiscriminatorMetric` to create any flavour of discriminator metric, e.g. NSGAN, WGAN, LSGAN etc.

    See :footcite:t:`lucic2018gans` for a comparison.

    :param torch.nn.Module metric: loss with which to compare outputs, defaults to :class:`torch.nn.MSELoss`
    :param float real_label: value for ideal real image, defaults to 1.
    :param float fake_label: value for ideal fake image, defaults to 0.
    :param bool no_grad: whether to no_grad the metric computation, defaults to ``False``
    :param str device: torch device, defaults to ``"cpu"``


    """

    def __init__(
        self,
        metric: nn.Module | None = None,
        real_label: float = 1.0,
        fake_label: float = 0.0,
        no_grad: bool = False,
        device="cpu",
    ):
        if metric is None:
            metric = nn.MSELoss()
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
        target = self.real if real else self.fake
        target = target.expand_as(pred) if pred.dim() > 0 else target

        with torch.no_grad() if self.no_grad else nullcontext():
            return self.metric(pred, target)


class AdversarialLoss(Loss):
    r"""Base adversarial loss.

    Override the forward function to call the adversarial loss with quantities depending on your specific GAN model.
    For examples, see :class:`deepinv.loss.adversarial.SupAdversarialLoss`
    and :class:`deepinv.loss.adversarial.UnsupAdversarialLoss`.

    See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for formulae.

    :param float weight_adv: weight for adversarial loss, defaults to 0.5.
    :param torch.nn.Module D: discriminator network. If not specified, `D` must be provided in forward(), defaults to None.
    :param torch.optim.Optimizer optimizer_D: optimizer for training discriminator.
        If `None` (default), do not train discriminator model.
    :param torch.optim.lr_scheduler.LRScheduler scheduler_D: optional learning rate scheduler
        for discriminator. If optimizer not passed, then this is ignored.
    :param str device: torch device, defaults to `"cpu"`
    """

    def __init__(
        self,
        weight_adv: float = 0.5,
        D: nn.Module = None,
        metric_gan: DiscriminatorMetric = None,
        optimizer_D: torch.optim.Optimizer = None,
        scheduler_D: torch.optim.lr_scheduler.LRScheduler = None,
        num_D_steps: int = 1,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metric_gan = (
            DiscriminatorMetric(device=device) if metric_gan is None else metric_gan
        )
        self.weight_adv = weight_adv
        self.D = D
        self.optimizer_D = optimizer_D
        self.scheduler_D = scheduler_D
        self.num_D_steps = num_D_steps
        if optimizer_D is None and scheduler_D is not None:
            raise ValueError(
                "Discriminator scheduler requires discriminator optimizer to be passed."
            )

        self.log_loss_D_train = AverageMeter("Training discrim loss", ":.2e")
        self.log_loss_D_eval = AverageMeter("Validation discrim loss", ":.2e")
        self.device = device

    def adversarial_gen(self, real: Tensor, fake: Tensor) -> torch.Tensor:
        r"""Adversarial penalty mechanism in GAN generators.

        :param torch.Tensor real: image labelled as real, typically one originating from training set
        :param torch.Tensor fake: image labelled as fake, typically a reconstructed image
        :return: generator adversarial loss
        """
        pred_fake = self.D(fake)
        return self.metric_gan(pred_fake, real=True) * self.weight_adv

    def adversarial_discrim(self, real: Tensor, fake: Tensor):
        r"""Adversarial penalty mechanism in GAN discriminators.

        :param torch.Tensor real: image labelled as real, typically one originating from training set
        :param torch.Tensor fake: image labelled as fake, typically a reconstructed image
        :return: (:class:`torch.Tensor`) discriminator adversarial loss
        """
        pred_real = self.D(real)
        pred_fake = self.D(fake.detach())

        adv_loss_real = self.metric_gan(pred_real, real=True)
        adv_loss_fake = self.metric_gan(pred_fake, real=False)

        return (adv_loss_real + adv_loss_fake) * self.weight_adv

    @contextmanager
    def step_discrim(self, model: nn.Module = None):
        """
        Context manager that steps discriminator optimizer
        that wraps a loss calculation.

        If discriminator optimizer does not exist, then this does nothing.

        :param torch.nn.Module model: generator model, used to detect if the loss is being
            used in training or evaluation mode. If it is in evaluation mode, then this
            function does nothing.
        """
        if self.optimizer_D is None:
            yield lambda loss: None
            return

        if model.training:
            self.optimizer_D.zero_grad()
        try:

            def backward(loss: torch.Tensor):
                if model.training:
                    self.log_loss_D_train.update(loss.item())
                    loss.backward(retain_graph=True)
                else:
                    self.log_loss_D_eval.update(loss.item())

            yield backward
        finally:
            if model.training:
                self.optimizer_D.step()
                if self.scheduler_D is not None:
                    self.scheduler_D.step()

    def forward(
        self,
        x_net: Tensor,
        x: Tensor,
        y: Tensor,
        physics: Physics,
        model: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full forward pass. Must calculate loss for discriminator, step the
        discriminator optimizer, then return the generator loss. For example:

        ::

            with self.step_discrim() as step:
                loss_d = ...
                step(loss_d)

            loss_g = ...
            return loss_g
        """
        raise NotImplementedError()

    def load_model(self, filename, device=None, strict: bool = True) -> dict:
        """Load discriminator from checkpoint.

        :param str, pathlib.Path filename: checkpoint filename.
        :param torch.device, str device: device to load model onto.
        :param bool strict: strict load weights to model.
        """
        ckpt = torch.load(filename, map_location=device, weights_only=False)
        if "state_dict" in ckpt:
            self.D.load_state_dict(ckpt["state_dict"], strict=strict)
        else:
            self.D.load_state_dict(ckpt, strict=strict)

        if "optimizer" in ckpt and self.optimizer_D is not None:
            self.optimizer_D.load_state_dict(ckpt["optimizer"])
        return ckpt

    def save_model(self, filename: str | Path):
        r"""
        Save the discriminator.

        :param str, pathlib.Path filename: filename to save to
        """
        torch.save(
            {
                "state_dict": self.D.state_dict(),
                "optimizer": (
                    self.optimizer_D.state_dict() if self.optimizer_D else None
                ),
            },
            filename,
        )
