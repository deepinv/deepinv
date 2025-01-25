from __future__ import annotations
from dataclasses import dataclass
from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

import torch
from torch.nn import Module

from deepinv.training.trainer import Trainer
from deepinv.loss import Loss
from deepinv.utils import AverageMeter


class AdversarialOptimizer:
    r"""
    Optimizer for adversarial training that encapsulates both generator and discriminator's optimizers.

    :param torch.optim.Optimizer optimizer_g: generator's torch optimizer
    :param torch.optim.Optimizer optimizer_d: discriminator's torch optimizer
    :param bool zero_grad_g_only: whether to only zero_grad generator, defaults to ``False``
    :param bool zero_grad_d_only: whether to only zero_grad discriminator, defaults to ``False``
    """

    def __init__(
        self,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        zero_grad_g_only: bool = False,
        zero_grad_d_only: bool = False,
    ):
        self.G = optimizer_g
        self.D = optimizer_d
        if zero_grad_d_only and zero_grad_g_only:
            raise ValueError("zero_grad_d_only or zero_grad_d_only must be False")
        self.zero_grad_d_only = zero_grad_d_only
        self.zero_grad_g_only = zero_grad_g_only

    def state_dict(self, *args, **kwargs):
        r"""Return both generator and discriminator's state_dicts with keys "G" and "D"."""
        return {"G": self.G.state_dict(), "D": self.D.state_dict()}

    def load_state_dict(self, state_dict):
        r"""Load state_dict which must have "G" and "D" keys for generator and discriminator respectively

        :param dict state_dict: state_dict with keys "G" and "D".
        """
        self.G.load_state_dict(state_dict["G"])
        self.D.load_state_dict(state_dict["D"])

    def zero_grad(self, set_to_none: bool = True):
        r"""zero_grad generator and discriminator optimizers, optionally only zero_grad one of them.

        :param bool set_to_none: whether to set gradients to None, defaults to True
        """
        if not self.zero_grad_d_only:
            self.G.zero_grad(set_to_none=set_to_none)
        if not self.zero_grad_g_only:
            self.D.zero_grad(set_to_none=set_to_none)


class AdversarialScheduler:
    r"""Scheduler for adversarial training that encapsulates both generator and discriminator's schedulers.

    :param LRScheduler scheduler_g: generator's torch scheduler
    :param LRScheduler scheduler_d: discriminator's torch scheduler
    """

    def __init__(self, scheduler_g: LRScheduler, scheduler_d: LRScheduler):
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d

    def get_last_lr(self):
        r"""Get last learning rates from the generator scheduler."""
        return self.scheduler_g.get_last_lr()

    def step(self):
        r"""Performs a step on both generator and discriminator schedulers."""
        self.scheduler_g.step()
        self.scheduler_d.step()


@dataclass
class AdversarialTrainer(Trainer):
    r"""AdversarialTrainer(model, physics, optimizer, train_dataloader, losses_d, D, step_ratio_D, ...)
    Trainer class for training a reconstruction network using adversarial learning.

    It overrides the :class:`deepinv.Trainer` class to provide the same functionality,
    whilst supporting training using adversarial losses. Note that the forward pass remains the same.

    The usual reconstruction model corresponds to the generator model in an adversarial framework,
    which is trained using losses specified in the ``losses`` argument.
    Additionally, a discriminator model ``D`` is also jointly trained using the losses provided in ``losses_d``.
    The adversarial losses themselves are defined in the :ref:`adversarial-losses` module.
    Examples of discriminators are in :ref:`adversarial`.

    See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for usage.

    |sep|

    :Examples:

        A very basic example:

        >>> from deepinv.training import AdversarialTrainer, AdversarialOptimizer
        >>> from deepinv.loss.adversarial import SupAdversarialGeneratorLoss, SupAdversarialDiscriminatorLoss
        >>> from deepinv.models import UNet, PatchGANDiscriminator
        >>> from deepinv.physics import LinearPhysics
        >>> from deepinv.datasets.utils import PlaceholderDataset
        >>>
        >>> generator = UNet(scales=2)
        >>> discrimin = PatchGANDiscriminator(1, 2, 1)
        >>>
        >>> optimizer = AdversarialOptimizer(
        ...     torch.optim.Adam(generator.parameters()),
        ...     torch.optim.Adam(discrimin.parameters()),
        ... )
        >>>
        >>> trainer = AdversarialTrainer(
        ...     model = generator,
        ...     D = discrimin,
        ...     physics = LinearPhysics(),
        ...     train_dataloader = torch.utils.data.DataLoader(PlaceholderDataset()),
        ...     epochs = 1,
        ...     losses = SupAdversarialGeneratorLoss(),
        ...     losses_d = SupAdversarialDiscriminatorLoss(),
        ...     optimizer = optimizer,
        ...     verbose = False
        ... )
        >>>
        >>> generator = trainer.train()


    Note that this forward pass also computes ``y_hat`` ahead of time to avoid having to compute it multiple times,
    but this is completely optional.

    See :class:`deepinv.Trainer` for additional parameters.

    :param deepinv.training.AdversarialOptimizer optimizer: optimizer encapsulating both generator and discriminator optimizers
    :param Loss, list losses_d: losses to train the discriminator, e.g. adversarial losses
    :param torch.nn.Module D: discriminator/critic/classification model, which must take in an image and return a scalar
    :param int step_ratio_D: every iteration, train D this many times, allowing for imbalanced generator/discriminator training. Defaults to 1.
    """

    optimizer: AdversarialOptimizer
    losses_d: Union[Loss, List[Loss]] = None
    D: Module = None
    step_ratio_D: int = 1

    def setup_train(self, **kwargs):
        r"""
        After usual Trainer setup, setup losses for discriminator too.
        """
        super().setup_train(**kwargs)

        if not isinstance(self.losses_d, (list, tuple)):
            self.losses_d = [self.losses_d]

        self.logs_losses_train += [
            AverageMeter("Training discrim loss " + l.name, ":.2e")
            for l in self.losses_d
        ]

        self.logs_losses_eval += [
            AverageMeter("Validation discrim loss " + l.name, ":.2e")
            for l in self.losses_d
        ]

        if self.ckpt_pretrained is not None:
            checkpoint = torch.load(self.ckpt_pretrained)
            self.D.load_state_dict(checkpoint["state_dict_D"])

        if self.check_grad:
            self.check_grad_val_D = AverageMeter(
                "Gradient norm for discriminator", ":.2e"
            )

    def compute_loss(self, physics, x, y, train=True, epoch: int = None):
        r"""
        Compute losses and perform backward passes for both generator and discriminator networks.

        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :returns: (tuple) The network reconstruction x_net (for plotting and computing metrics) and
            the logs (for printing the training progress).
        """
        logs = {}

        self.optimizer.G.zero_grad()

        # Evaluate reconstruction network
        x_net = self.model_inference(y=y, physics=physics)

        # Compute reconstructed measurement
        y_hat = physics.A(x_net)

        ### Train Generator
        if train or self.display_losses_eval:
            loss_total = 0
            for k, l in enumerate(self.losses):
                loss = l(
                    x=x,
                    x_net=x_net,
                    y=y,
                    y_hat=y_hat,
                    physics=physics,
                    model=self.model,
                    D=self.D,
                    epoch=epoch,
                )
                loss_total += loss.mean()
                if len(self.losses) > 1 and self.verbose_individual_losses:
                    current_log = (
                        self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                    )
                    current_log.update(loss.detach().cpu().numpy())
                    cur_loss = current_log.avg
                    logs[l.__class__.__name__] = cur_loss

            current_log = (
                self.logs_total_loss_train if train else self.logs_total_loss_eval
            )
            current_log.update(loss_total.item())

            logs[f"TotalLoss"] = current_log.avg

        if train:
            loss_total.backward(retain_graph=True)  # Backward the total generator loss

            norm = self.check_clip_grad()  # Optional gradient clipping
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg

            # Generator optimizer step
            self.optimizer.G.step()

        ### Train Discriminator
        for _ in range(self.step_ratio_D):
            if train or self.display_losses_eval:
                self.optimizer.D.zero_grad()

                loss_total_d = 0
                for k, l in enumerate(self.losses_d):
                    loss = l(
                        x=x,
                        x_net=x_net,
                        y=y,
                        y_hat=y_hat,
                        physics=physics,
                        model=self.model,
                        D=self.D,
                        epoch=epoch,
                    )
                    loss_total_d += loss.mean()
                    if len(self.losses_d) > 1 and self.verbose_individual_losses:
                        current_log = (
                            self.logs_losses_train[k + len(self.losses)]
                            if train
                            else self.logs_losses_eval[k + len(self.losses)]
                        )
                        current_log.update(loss.detach().cpu().numpy())
                        cur_loss = current_log.avg
                        logs[l.__class__.__name__] = cur_loss

            if train:
                loss_total_d.backward()

                norm = self.check_clip_grad_D()
                if norm is not None:
                    logs["gradient_norm_D"] = self.check_grad_val_D.avg

                self.optimizer.D.step()

        return x_net, logs

    def check_clip_grad_D(self):
        r"""Check the discriminator's gradient norm and perform gradient clipping if necessary.

        Analogous to ``check_clip_grad`` for generator.
        """
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)

        if self.check_grad:
            grads = [
                param.grad.detach().flatten()
                for param in self.D.parameters()
                if param.grad is not None
            ]
            norm_grads = torch.cat(grads).norm()
            self.check_grad_val_D.update(norm_grads.item())
            return norm_grads.item()

    def save_model(self, epoch, eval_psnr=None):
        r"""Save discriminator model parameters alongside other models."""
        super().save_model(epoch, eval_psnr, {"state_dict_D": self.D.state_dict()})
