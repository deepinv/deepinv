# RecoveryCoefficient

### *class* deepinv.loss.metric.RecoveryCoefficient(eps=None, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Recovery Coefficient metric used in emission tomography.

Computes the ratio between the total reconstructed activity and the total
ground-truth activity inside a given region of interest, defined by a mask:

$$
RC = \frac{\sum_i \hat{x}_i m_i}
          {\sum_i x_i m_i + \varepsilon}
$$

where $\hat{x}$ is the reconstructed image, $x$ is the ground-truth
image, $m$ is a binary or weighted mask defining the region of interest,
and $\varepsilon$ is a small constant added for numerical stability.

A value of `1` indicates perfect recovery of activity within the masked region.
Values below `1` indicate underestimation, while values above `1` indicate
overestimation.

#### NOTE
This metric requires a `mask` keyword argument to be provided during
evaluation.

#### NOTE
Higher values are better when they are closer to `1`. Therefore,
`lower_better=False`.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import RecoveryCoefficient
>>> m = RecoveryCoefficient()
>>> x = torch.ones(2, 1, 4, 4)
>>> x_net = torch.ones(2, 1, 4, 4)
>>> mask = torch.ones_like(x)
>>> m(x_net, x, mask=mask)
tensor([1., 1.])
```

* **Parameters:**
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Small constant added to the denominator for numerical stability.
  * **kwargs** – Additional keyword arguments passed to the parent
    [`deepinv.loss.metric.Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) class.

#### invert_metric(m)

“Invert metric for use as a training loss.
Recovery Coefficient is optimal at 1 so a sign flip does not produce a valid
loss
:param torch.Tensor m: calculated metric
