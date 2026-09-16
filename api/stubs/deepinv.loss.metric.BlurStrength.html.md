# BlurStrength

### *class* deepinv.loss.metric.BlurStrength(h_size=11, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

No-reference blur strength metric for batched images.

Returns a value in (0, 1) for each image in the batch, where 0 indicates a very sharp image and 1 indicates a very blurry image.

The metric has been introduced in Crete *et al.* [[32](https://deepinv.org/user_guide/other/biblio.html.md#id105)].

* **Parameters:**
  * **h_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – size of the uniform blur filter. Default: 11.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by ${\ell}_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

<hr />

* **Example:**

```pycon
>>> from deepinv.loss.metric import BlurStrength
>>> m = BlurStrength()
>>> x_net = torch.randn(2, 3, 16, 16)  # batch of 2 RGB images
>>> m(x_net).shape
torch.Size([2])
```

#### metric(x_net, \*args, \*\*kwargs)

Compute blur strength metric for a batch of images.

* **Parameters:**
  **x_net** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – (B, C, …) input tensors with C=1 or 3 channels. The spatial dimensions can be 1D, 2D, or higher.
* **Returns:**
  (B,) tensor of blur strength values in (0,1) for each image in the batch.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* sobel1d(x, axis)

Batched 1D Sobel derivative along an arbitrary axis.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(B, C, ...)`
  * **axis** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – axis along which to compute sobel derivative along.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `(B, C, ...)`
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* uniform_filter1d(x, size, axis)

Batched 1D uniform filter along an arbitrary axis.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape `(B, C, ...)`
  * **size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – size of filter
  * **axis** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – axis along which to compute filter
* **Returns:**
  filtered tensor of shape `(B, C, ...)`
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-loss-metric-blurstrength"></a>

## Examples using `BlurStrength`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div>
<!-- thumbnail-parent-div-close --></div>
