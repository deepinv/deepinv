from typing import List
from deepinv.loss.loss import Loss
from torch import Generator, randint

# TODO
# add test
# add example in docstring + add to docs
# add epoch in trainer loss call


class BaseLossScheduler(Loss):
    def __init__(self, *loss: Loss, generator=Generator()):
        super().__init__()
        self.losses = loss
        self.rng = generator

    def schedule(self, epoch) -> List[Loss]:
        return self.losses

    def forward(self, x_net, x, y, physics, model, epoch=None, **kwargs):
        losses = self.schedule(epoch)
        loss_total = 0
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
        return loss_total

    def adapt_model(self, model, **kwargs):
        for l in self.losses:
            model = l.adapt_model(model, **kwargs)
        return model


class RandomLossScheduler(BaseLossScheduler):
    def schedule(self, epoch) -> List[Loss]:
        choice = randint(2, (1,), generator=self.rng).item()
        return [self.losses[choice]]


class InterleavedLossScheduler(BaseLossScheduler):
    def __init__(self, *loss: Loss, generator=Generator()):
        super().__init__(*loss, generator=generator)
        self.choice = 0

    def schedule(self, epoch) -> List[Loss]:
        out = [self.losses[self.choice]]
        self.choice = (self.choice + 1) % len(self.losses)
        return out
