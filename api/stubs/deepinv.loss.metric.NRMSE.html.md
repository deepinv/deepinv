# NRMSE

### *class* deepinv.loss.metric.NRMSE(method='l2', \*\*kwargs)

Bases: [`NMSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NMSE.html.md#deepinv.loss.metric.NMSE)

Normalized Root Mean Squared Error metric.

Calculates

$$
\operatorname{NRMSE}(\hat{x},x)
= \frac{\|\hat{x}-x\|_2}{\|x\|_2}
= \sqrt{\operatorname{NMSE}(\hat{x},x)},
$$

where $\hat{x}=\inverse{y}$.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import NRMSE
>>> m = NRMSE()
>>> x_net = x = torch.ones(3, 2, 8, 8)
>>> m(x_net, x)
tensor([0., 0., 0.])
```

* **Parameters:**
  * **method** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalisation method. Currently only supports `l2`.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – method used to reduce scores over the batch dimension.
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – optional input normalization.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – optional spatial center crop.

<a id="sphx-glr-backref-deepinv-loss-metric-nrmse"></a>

## Examples using `NRMSE`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div>
<!-- thumbnail-parent-div-close --></div>
