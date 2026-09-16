# ERGAS

### *class* deepinv.loss.metric.ERGAS(factor, \*args, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Error relative global dimensionless synthesis metric.

Calculates the ERGAS metric on a multispectral image and a target.
ERGAS is a popular metric for pan-sharpening of multispectral images.

Wraps the `torchmetrics` [ERGAS](https://lightning.ai/docs/torchmetrics/stable/image/error_relative_global_dimensionless_synthesis.html) function.
Note that our `reduction` parameter follows our uniform convention (see below).

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import ERGAS
>>> m = ERGAS(factor=4)
>>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
>>> m(x_net, x)
tensor([0., 0., 0.])
```

* **Parameters:**
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – pansharpening factor.
  * **train_loss** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use metric as a training loss, by returning one minus the metric.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

<a id="sphx-glr-backref-deepinv-loss-metric-ergas"></a>

## Examples using `ERGAS`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div>
<!-- thumbnail-parent-div-close --></div>
