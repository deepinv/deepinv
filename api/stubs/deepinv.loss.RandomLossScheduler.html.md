# RandomLossScheduler

### *class* deepinv.loss.RandomLossScheduler(\*loss, generator=None, weightings=None)

Bases: [`BaseLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.BaseLossScheduler.html.md#deepinv.loss.BaseLossScheduler)

Schedule losses at random.

The scheduler wraps a list of losses. Each time this is called, one loss is selected at random and used for the forward pass.

Optionally pass a weighting for each loss e.g. if `weightings=[3, 1]` then the first loss is 3 times more likely to be called than the second loss.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss import RandomLossScheduler, SupLoss
>>> from deepinv.loss.metric import SSIM
>>> l = RandomLossScheduler(SupLoss(), SSIM(train_loss=True)) # Choose randomly between Sup and SSIM
>>> x_net = x = torch.tensor([0., 0., 0.])
>>> l(x=x, x_net=x_net)
tensor(0.)
```

* **Parameters:**
  * **\*loss** ([*Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)) – loss or multiple losses to be scheduled.
  * **generator** (*Generator*) – torch random number generator, defaults to None
