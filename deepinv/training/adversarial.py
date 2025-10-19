from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tqdm

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

import warnings

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

    def state_dict(self):
        r"""Returns the state of both schedulers as a dictionary."""
        return {
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
        }

    def load_state_dict(self, state_dict):
        r"""Loads the state of both schedulers from a dictionary."""
        self.scheduler_g.load_state_dict(state_dict["scheduler_g"])
        self.scheduler_d.load_state_dict(state_dict["scheduler_d"])


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
        ...     verbose = False,
        ...     device = "cpu",
        ...     optimizer_step_multi_dataset = False
        ... )
        >>>
        >>> generator = trainer.train()


    Note that this forward pass also computes ``y_hat`` ahead of time to avoid having to compute it multiple times,
    but this is completely optional.

    See :class:`deepinv.Trainer` for additional parameters.

    .. warning::

        The multi-dataset option is not available yet when using an adversarial trainer. The `optimizer_step_multi_dataset` parameter is therefore automatically set to `False` if not set to `False` by the user.


    :param deepinv.training.AdversarialOptimizer optimizer: optimizer encapsulating both generator and discriminator optimizers
    :param Loss, list losses_d: losses to train the discriminator, e.g. adversarial losses
    :param torch.nn.Module D: discriminator/critic/classification model, which must take in an image and return a scalar
    :param int step_ratio_D: every iteration, train D this many times, allowing for imbalanced generator/discriminator training. Defaults to 1.
    """

    optimizer: AdversarialOptimizer
    losses_d: Loss | list[Loss] = None
    D: Module = None
    step_ratio_D: int = 1

    def setup_run(self, **kwargs):
        r"""
        After usual Trainer setup, setup losses for discriminator too.
        """
        self.epoch_start = 0

        super()._setup_data()
        super()._setup_logging()

        if self.optimizer_step_multi_dataset:
            warnings.warn(
                "optimizer_step_multi_dataset parameter of Trainer should be set to `False` when using adversarial trainer. Automatically setting it to `False`."
            )
            self.optimizer_step_multi_dataset = False

        if not isinstance(self.losses_d, (list, tuple)):
            self.losses_d = [self.losses_d]

        for l in self.losses_d:
            self.meters_losses_train[l.__class__.__name__] = AverageMeter(
                "Training discrim loss " + l.__class__.__name__, ":.2e"
            )
            self.meters_losses_val[l.__class__.__name__] = AverageMeter(
                "Validation discrim loss " + l.__class__.__name__, ":.2e"
            )

        if self.log_grad:
            self.check_grad_val_D = AverageMeter(
                "Gradient norm for discriminator", ":.2e"
            )

        if self.ckpt_pretrained is not None:
            self.load_ckpt(self.ckpt_pretrained)
            checkpoint = torch.load(self.ckpt_pretrained)
            self.D.load_state_dict(checkpoint["state_dict_D"])

    def _compute_losses(
        self,
        losses,
        x,
        x_net,
        y,
        physics,
        train=True,
        epoch: int = None,
    ):
        r"""
        Compute losses for a given set of loss functions.

        :param list losses: List of loss functions to compute.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :returns: Total loss value.
        """
        y_hat = physics.A(x_net)

        with torch.set_grad_enabled(train):
            loss_total = 0
            for l in losses:
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

                if len(losses) > 1:
                    meter = (
                        self.meters_losses_train[l.__class__.__name__]
                        if train
                        else self.meters_losses_val[l.__class__.__name__]
                    )
                    meter.update(loss.detach().cpu().numpy())

            meter = self.meter_total_loss_train if train else self.meter_total_loss_val
            meter.update(loss_total.item())

        return loss_total

    def compute_losses_generator(
        self,
        x,
        x_net,
        y,
        physics,
        train=True,
        epoch: int = None,
    ):
        r"""
        Compute losses for the generator network.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :returns: Total generator loss.
        """
        return self._compute_losses(self.losses, x, x_net, y, physics, train, epoch)

    def compute_losses_discriminator(
        self,
        x,
        x_net,
        y,
        physics,
        train=True,
        epoch: int = None,
    ):
        r"""
        Compute losses for the discriminator network.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :returns: Total discriminator loss.
        """
        return self._compute_losses(self.losses_d, x, x_net, y, physics, train, epoch)

    def step(
        self,
        epoch: int,
        progress_bar: tqdm,
        train_ite: int = None,
        phase: str = "train",
        last_batch: bool = False,
    ) -> None:
        r"""
        Train/Eval a batch.

        It performs the forward pass, the backward pass, and the evaluation at each iteration
        for both generator and discriminator networks.

        :param int epoch: Current epoch.
        :param progress_bar: `tqdm <https://tqdm.github.io/docs/tqdm/>`_ progress bar.
        :param int train_ite: train iteration, only needed for logging if ``Trainer.log_train_batch=True``
        :param str phase: Training phase ('train', 'val', 'test').
        :param bool last_batch: If ``True``, the last batch of the epoch is being processed.
        """
        if phase == "train":
            training_step = True
        else:
            training_step = False

        ### Train Generator

        # Zero grad
        if training_step:
            self.optimizer.zero_grad()

        # Get either online or offline samples
        x, y, physics_cur = self.get_samples(
            (
                self.current_train_iterators
                if training_step
                else self.current_val_iterators
            ),
            0,
        )

        # Evaluate reconstruction network
        x_net = self.model_inference(y=y, physics=physics_cur, x=x, train=training_step)

        # Compute the loss for the batch
        loss_generator_cur = self.compute_losses_generator(
            x=x,
            x_net=x_net,
            y=y,
            physics=physics_cur,
            train=training_step,
            epoch=epoch,
        )

        # Backward + Optimizer
        if training_step:
            loss_generator_cur.backward(retain_graph=True)

        if training_step:
            loss_logs = {}
            loss_logs["Loss Generator"] = loss_generator_cur.item()

            # Gradient clipping
            grad_norm = self.apply_grad_clip()
            if self.log_grad:
                loss_logs["gradient_norm"] = grad_norm

            # Optimizer step
            self.optimizer.G.step()

            # Update the progress bar
            progress_bar.set_postfix(loss_logs)

        ### Train Discriminator
        for _ in range(self.step_ratio_D):

            if training_step:
                self.optimizer.D.zero_grad()

            # Compute the discriminator loss for the batch
            loss_discriminator_cur = self.compute_losses_discriminator(
                x,
                x_net,
                y,
                physics_cur,
                train=training_step,
                epoch=epoch,
            )

            # Backward + Optimizer
            if training_step:
                loss_discriminator_cur.backward()

                loss_logs = {}
                loss_logs["Loss Discriminator"] = loss_discriminator_cur.item()

                # Gradient clipping
                norm = self.apply_grad_clip_D()
                if norm is not None:
                    loss_logs["gradient_norm_D"] = self.check_grad_val_D.avg

                self.optimizer.D.step()

        # Compute the metrics for the batch
        x_net = x_net.detach()  # detach the network output for metrics and plotting
        self.compute_metrics(x, x_net, y, physics_cur, train=training_step, epoch=epoch)

        # Log images of last batch for each dataset
        if last_batch:
            self.save_images(
                epoch,
                physics_cur,
                x,
                y,
                x_net,
                train=training_step,
            )

        # Log epoch losses and metrics
        if last_batch:

            ## Losses
            epoch_loss_logs = {}

            # Add individual losses over an epoch
            if len(self.losses) > 1:
                for l in self.losses:
                    meter = (
                        self.meters_losses_train[l.__class__.__name__]
                        if training_step
                        else self.meters_losses_val[l.__class__.__name__]
                    )
                    epoch_loss_logs[l.__class__.__name__] = meter.avg

            # Add total loss over an epoch
            meter = (
                self.meter_total_loss_train
                if training_step
                else self.meter_total_loss_val
            )
            epoch_loss_logs["Total_Loss"] = meter.avg

            ## Metrics
            epoch_metrics_logs = {}
            for m in self.metrics:
                meter = (
                    self.meters_metrics_train[m.__class__.__name__]
                    if training_step
                    else self.meters_metrics_val[m.__class__.__name__]
                )
                epoch_metrics_logs[m.__class__.__name__] = meter.avg

            ## Logging
            for logger in self.loggers:
                logger.log_losses(epoch_loss_logs, step=epoch, phase=phase)
                logger.log_metrics(epoch_metrics_logs, step=epoch, phase=phase)

    def apply_grad_clip_D(self):
        r"""Check the discriminator's gradient norm and perform gradient clipping if necessary.

        Analogous to ``check_clip_grad`` for generator.
        """
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)

        if self.log_grad:
            grads = [
                param.grad.detach().flatten()
                for param in self.D.parameters()
                if param.grad is not None
            ]
            norm_grads = torch.cat(grads).norm()
            self.check_grad_val_D.update(norm_grads.item())
            return norm_grads.item()

    def save_ckpt(self, epoch: int, name: str | None = None) -> None:
        r"""
        Save necessary information to resume training.

        :param int epoch: Current epoch.
        :param str name: Name of the checkpoint file.
        """
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "state_dict_D": self.D.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "loss": self.train_loss_history,
            "val_metrics": self.val_metrics_history_per_epoch,
        }

        for logger in self.loggers:
            logger.log_checkpoint(epoch=epoch, state=state, name=name)
