# LpNorm

### *class* deepinv.loss.metric.LpNorm(p=2, onesided=False, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

$\ell_p$ metric for $p>0$.

Calculates $L_p(\hat{x},x)$ where $\hat{x}=\inverse{y}$.

If `onesided=False` then the metric is defined as
$d(x,y)=\|x-y\|_p^p$.

Otherwise, it is the one-sided error Jacques *et al.*<sup>[1](#footcite-jacques2013robust)</sup>, defined as
$d(x,y)= \|\max(x\circ y) \|_p^p$. where $\circ$ denotes element-wise multiplication.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import LpNorm
>>> m = LpNorm(p=3) # L3 norm
>>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
>>> m(x_net, x)
tensor([0., 0., 0.])
```

* **Parameters:**
  * **p** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – order p of the Lp norm
  * **onesided** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether one-sided metric.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

<hr />

* **References:**

* <a id='footcite-jacques2013robust'>**[1]**</a> Laurent Jacques, Jason N Laska, Petros T Boufounos, and Richard G Baraniuk. Robust 1-bit compressive sensing via binary stable embeddings of sparse vectors. *IEEE transactions on information theory*, 59(4):2082–2102, 2013.
