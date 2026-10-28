# InterleavedEpochLossScheduler

### *class* deepinv.loss.InterleavedEpochLossScheduler(\*loss, generator=None)

Bases: [`BaseLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.BaseLossScheduler.html.md#deepinv.loss.BaseLossScheduler)

Schedule losses sequentially epoch-by-epoch.

The scheduler wraps a list of losses. Each epoch, the next loss is selected in order and used for the forward pass for that epoch.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss import InterleavedEpochLossScheduler, SupLoss
>>> from deepinv.loss.metric import SSIM
>>> l = InterleavedEpochLossScheduler(SupLoss(), SSIM(train_loss=True)) # Choose alternating between Sup and SSIM
>>> x_net = x = torch.tensor([0., 0., 0.])
>>> l(x=x, x_net=x_net, epoch=0)
tensor(0.)
```

* **Parameters:**
  **\*loss** ([*Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)) – loss or multiple losses to be scheduled.
