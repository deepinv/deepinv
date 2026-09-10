# QNR

### *class* deepinv.loss.metric.QNR(alpha=1, beta=1, p=1, q=1, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Quality with No Reference (QNR) metric for pansharpening.

Calculates the no-reference $\text{QNR}(\hat{x})$ where $\hat{x}=\inverse{y}$.

QNR was proposed in Alparone et al., “Multispectral and Panchromatic Data Fusion Assessment Without Reference”.

Note we don’t use the torchmetrics implementation.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import QNR
>>> from deepinv.physics import Pansharpen
>>> m = QNR()
>>> x = x_net = torch.rand(1, 3, 64, 64) # B,C,H,W
>>> physics = Pansharpen((3, 64, 64))
>>> y = physics(x) #[BCH'W', B1HW]
>>> m(x_net=x_net, y=y, physics=physics)
tensor([...])
```

* **Parameters:**
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – weight for spectral quality, defaults to 1
  * **beta** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – weight for structural quality, defaults to 1
  * **p** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – power exponent for spectral D, defaults to 1
  * **q** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – power exponent for structural D, defaults to 1
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **train_loss** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use metric as a training loss, by returning one minus the metric.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

#### D_lambda(hrms, lrms)

Calculate spectral distortion index.

#### D_s(hrms, lrms, pan, pan_lr)

Calculate spatial (or structural) distortion index.

#### metric(x_net, x, y, physics, \*args, \*\*kwargs)

Calculate QNR on data.

#### NOTE
Note this does not require knowledge of `x`, but it is included here as a placeholder.

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reconstructed high-res multispectral image $\inverse{y}$ of shape `(B,C,H,W)`.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Placeholder, does nothing.
  * **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)) – pansharpening measurements generated from
    [`deepinv.physics.Pansharpen`](https://deepinv.org/api/stubs/deepinv.physics.Pansharpen.html.md#deepinv.physics.Pansharpen), where y[0] is the low-res multispectral image of shape `(B,C,H',W')`
    and y[1] is the high-res noisy panchromatic image of shape `(B,1,H,W)`
  * **physics** ([*deepinv.physics.Pansharpen*](https://deepinv.org/api/stubs/deepinv.physics.Pansharpen.html.md#deepinv.physics.Pansharpen)) – pansharpening physics, used to calculate low-res pan image for QNR calculation.
* **Return torch.Tensor:**
  calculated metric, the tensor size might be `(1,)` or `(B,)`.
