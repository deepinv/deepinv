# Metric

### *class* deepinv.loss.metric.Metric(metric=None, complex_abs=False, train_loss=False, reduction=None, norm_inputs=None, center_crop=None)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class for metrics.

See docs for [`forward`](#deepinv.loss.metric.Metric.forward) below for more details.

To create a new metric, inherit from this class, override the [`metric method`](#deepinv.loss.metric.Metric.metric),
set `lower_better` attribute and optionally override the `invert_metric` method.

You can also directly use this baseclass to wrap an existing metric function, e.g. from
[torchmetrics](https://lightning.ai/docs/torchmetrics/stable), to benefit from our preprocessing.
The metric function must reduce over all dims except the batch dim (see example).

* **Parameters:**
  * **metric** (*Callable*) – metric function, it must reduce over all dims except batch dim. It must not reduce over batch dim.
    This is unused if the `metric` method is overridden. Takes as input `x_net` and `x` tensors and returns a tensor of metric scores.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **train_loss** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if higher is better, invert metric. If lower is better, does nothing.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input, `clip` clips to $[0,1]$, `standardize` standardizes to same mean and std as ground truth, `none` or None no normalization will be applied (default).
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

<hr />

### Examples

Use `Metric` to wrap functional metrics such as from torchmetrics:

```pycon
>>> from functools import partial
>>> from torchmetrics.functional.image import structural_similarity_index_measure
>>> from deepinv.loss.metric import Metric
>>> m = Metric(metric=partial(structural_similarity_index_measure, reduction='none'))
>>> x = x_net = torch.ones(2, 3, 64, 64) # B,C,H,W
>>> m(x_net - 0.1, x)
tensor([0., 0.])
```

#### \_\_add_\_(other)

Sums two metrics via the + operation.

* **Parameters:**
  **other** ([*deepinv.loss.metric.Metric*](#deepinv.loss.metric.Metric)) – other metric
* **Returns:**
  [`deepinv.loss.metric.Metric`](#deepinv.loss.metric.Metric) summed metric.
* **Return type:**
  [*Metric*](#deepinv.loss.metric.Metric)

#### forward(x_net=None, x=None, \*args, \*\*kwargs)

Metric forward pass.

Usually, the data passed is `x_net, x` i.e. estimate and target or only `x_net` for no-reference metric.

The forward pass also optionally calculates complex magnitude of images, performs normalisation,
or inverts the metric to use it as a training loss (if by default higher is better).

By default, no reduction is performed in the batch dimension, but mean or sum reduction can be performed too.

All tensors should be of shape `(B, ...)` or `(B, C, ...)` where `B` is batch size and `C` is channels.

#### NOTE
If a full reference metric is used and a tensor is `None`, a tensor of NaN will be returned instead.

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reconstructed image $\hat{x}=\inverse{y}$ of shape `(B, ...)` or `(B, C, ...)`.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reference image $x$ (optional) of shape `(B, ...)` or `(B, C, ...)`.
* **Return torch.Tensor:**
  calculated metric, the tensor size might be `(1,)` or `(B,)`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### invert_metric(m)

Invert metric. Used where a higher=better metric is to be used in a training loss.

* **Parameters:**
  **m** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – calculated metric

#### metric(x_net=None, x=None, \*args, \*\*kwargs)

Calculate metric on data.

Override this function to implement your own metric. Always include `args` and `kwargs` arguments.
Do not perform reduction.

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reconstructed image $\hat{x}=\inverse{y}$ of shape `(B, ...)` or `(B, C, ...)`.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reference image $x$ (optional) of shape `(B, ...)` or `(B, C, ...)`.
* **Return torch.Tensor:**
  calculated unreduced metric of shape `(B,)`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
