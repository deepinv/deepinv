# HaarPSI

### *class* deepinv.loss.metric.HaarPSI(C=5.0, alpha=4.9, preprocess_with_subsampling=True, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

HaarPSI metric with tuned parameters.

The metric was proposed by [Reisenhofer et al.](https://arxiv.org/abs/1607.06140) and the parameters are taken from [Karner et al.](https://arxiv.org/abs/2410.24098).
The metric computes similarities in the Haar wavelet domain and it is shown to closely match human evaluation. See original papers for more details.
The metric range is $[0,1]$. The higher the metric, the better.

Code is adapted from [this implementation](https://github.com/ideal-iqa/haarpsi-pytorch) by Sören Dittmer, Clemens Karner and Anna Breger, adapted from David Neumann, adapted from Rafael Reisenhofer.

#### NOTE
Images must be scaled to $[0,1]$. You can use `norm_inputs = clip` or `min_max` to achieve this.

The parameters should be set as follows depending on the image domain:

- **Natural images**: $C=30,\alpha=4.2$.
- **Medical images**: $C=5,\alpha=4.9$.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import HaarPSI
>>> m = HaarPSI(norm_inputs="clip")
>>> x_net = x = torch.ones(3, 1, 8, 8) # B,C,H,W
>>> m(x_net, x)
tensor([1.0000, 1.0000, 1.0000])
```

* **Parameters:**
  * **C** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – metric parameter $C\in[5, 100]$.
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – metric paramter $\alpha\in[2, 8]$.
  * **preprocess_with_subsampling** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Determines if subsampling is performed.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
