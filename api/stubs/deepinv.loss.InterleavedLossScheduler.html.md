# InterleavedLossScheduler

### *class* deepinv.loss.InterleavedLossScheduler(\*loss)

Bases: [`BaseLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.BaseLossScheduler.html.md#deepinv.loss.BaseLossScheduler)

Schedule losses sequentially one-by-one.

The scheduler wraps a list of losses. Each time this is called, the next loss is selected in order and used for the forward pass.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss import InterleavedLossScheduler, SupLoss
>>> from deepinv.loss.metric import SSIM
>>> l = InterleavedLossScheduler(SupLoss(), SSIM(train_loss=True)) # Choose alternating between Sup and SSIM
>>> x_net = x = torch.tensor([0., 0., 0.])
>>> l(x=x, x_net=x_net)
tensor(0.)
```

* **Parameters:**
  **\*loss** ([*Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)) – loss or multiple losses to be scheduled.
