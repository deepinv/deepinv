from typing import List
from torch import Generator, randint, Tensor, tensor
from torch.nn import Module
from deepinv.loss.loss import Loss
from deepinv.physics.forward import Physics


class BaseLossScheduler(Loss):
    r"""
    Base class for loss schedulers.

    Wraps a list of losses, and each time forward is called, some of them are selected based on a defined schedule.

    :param Loss \*loss: loss or multiple losses to be scheduled.
    :param Generator generator: torch random number generator, defaults to None
    """

    def __init__(self, *loss: Loss, generator: Generator = None):
        super().__init__()
        self.losses = loss
        self.rng = generator if generator is not None else Generator()

    def schedule(self, epoch: int) -> List[Loss]:
        r"""
        Return selected losses based on defined schedule, optionally based on current epoch.

        :param int epoch: current epoch number
        :return list[Loss]: selected (sub)list of losses to be used this time.
        """
        return self.losses

    def forward(
        self,
        x_net: Tensor = None,
        x: Tensor = None,
        y: Tensor = None,
        physics: Physics = None,
        model: Module = None,
        epoch: int = None,
        **kwargs,
    ):
        r"""
        Loss forward pass.

        When called, subselect losses based on defined schedule to be used at this pass, and apply to inputs.

        :param torch.Tensor x_net: model output
        :param torch.Tensor x: ground truth
        :param torch.Tensor y: measurement
        :param Physics physics: measurement operator
        :param torch.nn.Module model: reconstruction model
        :param int epoch: current epoch
        """
        losses = self.schedule(epoch)
        loss_total = 0.0
        for l in losses:
            loss_total += l.forward(
                x_net=x_net,
                x=x,
                y=y,
                physics=physics,
                model=model,
                epoch=epoch,
                **kwargs,
            )
        if isinstance(loss_total, float):
            return tensor(loss_total, requires_grad=True)
        return loss_total

    def adapt_model(self, model: Module, **kwargs):
        r"""
        Adapt model using all wrapped losses.

        Some loss functions require the model forward call to be adapted before the forward pass.

        :param torch.nn.Module model: reconstruction model
        """
        for l in self.losses:
            model = l.adapt_model(model, **kwargs)
        return model


class RandomLossScheduler(BaseLossScheduler):
    r"""
    Schedule losses at random.

    The scheduler wraps a list of losses. Each time this is called, one loss is selected at random and used for the forward pass.

    :Example:

    >>> import torch
    >>> from deepinv.loss import RandomLossScheduler, SupLoss
    >>> from deepinv.loss.metric import SSIM
    >>> l = RandomLossScheduler(SupLoss(), SSIM(train_loss=True)) # Choose randomly between Sup and SSIM
    >>> x_net = x = torch.tensor([0., 0., 0.])
    >>> l(x=x, x_net=x_net)
    tensor(0.)

    :param Loss \*loss: loss or multiple losses to be scheduled.
    :param Generator generator: torch random number generator, defaults to None
    """

    def schedule(self, epoch) -> List[Loss]:
        choice = randint(2, (1,), generator=self.rng).item()
        return [self.losses[choice]]


class InterleavedLossScheduler(BaseLossScheduler):
    r"""
    Schedule losses sequentially one-by-one.

    The scheduler wraps a list of losses. Each time this is called, the next loss is selected in order and used for the forward pass.

    :Example:

    >>> import torch
    >>> from deepinv.loss import InterleavedLossScheduler, SupLoss
    >>> from deepinv.loss.metric import SSIM
    >>> l = InterleavedLossScheduler(SupLoss(), SSIM(train_loss=True)) # Choose alternating between Sup and SSIM
    >>> x_net = x = torch.tensor([0., 0., 0.])
    >>> l(x=x, x_net=x_net)
    tensor(0.)

    :param Loss \*loss: loss or multiple losses to be scheduled.
    """

    def __init__(self, *loss: Loss):
        super().__init__(*loss)
        self.choice = 0

    def schedule(self, epoch) -> List[Loss]:
        out = [self.losses[self.choice]]
        self.choice = (self.choice + 1) % len(self.losses)
        return out


class InterleavedEpochLossScheduler(BaseLossScheduler):
    r"""
    Schedule losses sequentially epoch-by-epoch.

    The scheduler wraps a list of losses. Each epoch, the next loss is selected in order and used for the forward pass for that epoch.

    :Example:

    >>> import torch
    >>> from deepinv.loss import InterleavedEpochLossScheduler, SupLoss
    >>> from deepinv.loss.metric import SSIM
    >>> l = InterleavedEpochLossScheduler(SupLoss(), SSIM(train_loss=True)) # Choose alternating between Sup and SSIM
    >>> x_net = x = torch.tensor([0., 0., 0.])
    >>> l(x=x, x_net=x_net, epoch=0)
    tensor(0.)

    :param Loss \*loss: loss or multiple losses to be scheduled.
    """

    def schedule(self, epoch) -> List[Loss]:
        return [self.losses[epoch % len(self.losses)]]


class StepLossScheduler(BaseLossScheduler):
    r"""
    Activate losses at specified epoch.

    The scheduler wraps a list of losses. When epoch is <= threshold, this returns 0. Otherwise, it returns the sum of the losses.

    :Example:

    >>> import torch
    >>> from deepinv.loss import StepLossScheduler
    >>> from deepinv.loss.metric import SSIM
    >>> l = StepLossScheduler(SSIM(train_loss=True)) # Use SSIM only after epoch 10
    >>> x_net = torch.zeros(1, 1, 12, 12)
    >>> x = torch.ones(1, 1, 12, 12)
    >>> l(x=x, x_net=x_net, epoch=0)
    tensor(0., requires_grad=True)
    >>> l(x=x, x_net=x_net, epoch=11)
    tensor([0.9999])

    :param Loss \*loss: loss or multiple losses to be scheduled.
    :param int epoch_thresh: threshold above which the losses are used.
    """

    def __init__(self, *loss: Loss, epoch_thresh: int = 0):
        super().__init__(*loss)
        self.epoch_thresh = epoch_thresh

    def schedule(self, epoch) -> List[Loss]:
        return self.losses if epoch > self.epoch_thresh else []
