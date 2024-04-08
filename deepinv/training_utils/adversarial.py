from dataclasses import dataclass
from typing import Union, List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module
from .trainer import Trainer
from deepinv.loss import Loss
from deepinv.utils import AverageMeter


class AdversarialOptimizer:
    def __init__(
        self,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        zero_grad_g_only=False,
        zero_grad_d_only=False,
    ):
        self.G = optimizer_g
        self.D = optimizer_d
        if zero_grad_d_only and zero_grad_g_only:
            raise ValueError("zero_grad_d_only or zero_grad_d_only must be False")
        self.zero_grad_d_only = zero_grad_d_only
        self.zero_grad_g_only = zero_grad_g_only

    def load_state_dict(self, state_dict):
        # TODO need to implement for both G and D
        # will need to also override the checkpoint loading and saving in Trainer
        return NotImplementedError()

    def state_dict(self):
        return self.G.state_dict()

    def zero_grad(self, set_to_none: bool = True):
        if not self.zero_grad_d_only:
            self.G.zero_grad(set_to_none=set_to_none)
        if not self.zero_grad_g_only:
            self.D.zero_grad(set_to_none=set_to_none)


class AdversarialScheduler:
    def __init__(self, scheduler_g: LRScheduler, scheduler_d: LRScheduler):
        self.G = scheduler_g
        self.D = scheduler_d

    def get_last_lr(self):
        return self.G.get_last_lr()

    def step(self):
        self.G.step()
        self.D.step()


@dataclass
class AdversarialTrainer(Trainer):
    """
    Notes
    - usual reconstruction model corresponds to the generator model
    - Forward pass remains same
    - Computes y_hat ahead of time (to avoid having to compute it in both G and D's losses) but not all adversarial losses may need this
    """

    optimizer: AdversarialOptimizer
    losses_d: Union[Loss, List[Loss]] = None
    D: Module = None

    def setup_train(self):
        """
        After usual Trainer setup, setup losses for discriminator too
        """
        super().setup_train()

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

    def compute_loss(self, physics, x, y, train=True):
        r"""
        Compute the loss and perform the backward pass.

        It evaluates the reconstruction network, computes the losses, and performs the backward pass.

        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :returns: (tuple) The network reconstruction x_net (for plotting and computing metrics) and
            the logs (for printing the training progress).
        """
        logs = {}

        # TODO is this right place?
        self.optimizer.G.zero_grad()

        # Evaluate reconstruction network
        x_net = self.model_inference(y=y, physics=physics)

        # Compute reconstructed measurement
        y_hat = physics.A(x_net)

        # Compute generator losses
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

        # Compute discriminator losses
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
            loss_total_d.backward()  # Backward the total discriminator loss

            # TODO discriminator gradient clipping

            # Discriminator optimizer step
            self.optimizer.D.step()

        return x_net, logs
