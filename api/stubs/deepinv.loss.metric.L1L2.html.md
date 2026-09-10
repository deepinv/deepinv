# L1L2

### *class* deepinv.loss.metric.L1L2(alpha=0.5, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Combined L2 and L1 metric.

Calculates L2 distance (i.e. MSE) + L1 (i.e. MAE) distance,
$\alpha L_1(\hat{x},x)+(1-\alpha)L_2(\hat{x},x)$ where $\hat{x}=\inverse{y}$.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import L1L2
>>> m = L1L2()
>>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
>>> m(x_net, x)
tensor([0., 0., 0.])
```

* **Parameters:**
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weight between L2 and L1. Defaults to 0.5.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
