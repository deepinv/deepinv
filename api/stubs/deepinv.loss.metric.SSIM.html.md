# SSIM

### *class* deepinv.loss.metric.SSIM(multiscale=False, max_pixel=1.0, min_pixel=0.0, torchmetric_kwargs=None, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Structural Similarity Index (SSIM) metric using torchmetrics.

Calculates $\text{SSIM}(\hat{x},x)$ where $\hat{x}=\inverse{y}$.
See [https://en.wikipedia.org/wiki/Structural_similarity](https://en.wikipedia.org/wiki/Structural_similarity) for more information.

To set the max pixel on the fly (as is the case in [fastMRI evaluation code](https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/common/evaluate.py)), set `max_pixel=None`.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import SSIM
>>> m = SSIM()
>>> x_net = x = torch.ones(3, 2, 32, 32) # B,C,H,W
>>> m(x_net, x)
tensor([1., 1., 1.])
```

* **Parameters:**
  * **multiscale** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, computes the multi-scale SSIM. Default: `False`.
  * **max_pixel** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum pixel value. If None, uses max pixel value of the ground truth image x.
  * **min_pixel** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – minimum pixel value. If None, uses min pixel value of the ground truth image x.
  * **torchmetric_kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – kwargs for torchmetrics SSIM as dict. See [https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html](https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html)
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **train_loss** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use metric as a training loss, by returning one minus the metric.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
