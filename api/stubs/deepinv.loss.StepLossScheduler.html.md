# StepLossScheduler

### *class* deepinv.loss.StepLossScheduler(\*loss, epoch_thresh=0)

Bases: [`BaseLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.BaseLossScheduler.html.md#deepinv.loss.BaseLossScheduler)

Activate losses at specified epoch.

The scheduler wraps a list of losses. When epoch is <= threshold, this returns 0. Otherwise, it returns the sum of the losses.

* **Example:**

```pycon
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
```

* **Parameters:**
  * **\*loss** ([*Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)) – loss or multiple losses to be scheduled.
  * **epoch_thresh** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – threshold above which the losses are used.
