# SharpnessIndex

### *class* deepinv.loss.metric.SharpnessIndex(periodic_component=True, dequantize=True, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

No-reference sharpness index metric for 2D images.

Measures how sharp an image is, defined as

$$
\text{SI}(x) = -\log \Phi \left( \frac{\mathbb{E}_{\omega} \{ \text{TV}(\omega * x)\} - \text{TV}(x)  }{\sqrt{\mathbb{V}_{\omega} \{ \text{TV}(\omega * x) \} } } \right)
$$

where $\Phi$ is the CDF of a standard Gaussian distribution, $\text{TV}$ is the total variation,
and $\omega \sim \mathcal{N}(0, I)$ is a Gaussian white noise distribution.

Higher values indicate sharper images.

The metric is used to introduced by Blanchet and Moisan [[12](https://deepinv.org/user_guide/other/biblio.html.md#id102)].
We use the fast implementation presented by Leclaire and Moisan [[77](https://deepinv.org/user_guide/other/biblio.html.md#id101)].

Adapted from MATLAB implementation in [https://helios2.mi.parisdescartes.fr/~moisan/sharpness/](https://helios2.mi.parisdescartes.fr/~moisan/sharpness/).

Default mode computing the periodic component and dequantizing should be used, unless you want to work on very
specific images that are naturally periodic or not quantized (see Leclaire and Moisan [[77](https://deepinv.org/user_guide/other/biblio.html.md#id101)]).

* **Parameters:**
  * **periodic_component** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True` (default), compute the periodic component of the image before computing the metric.
  * **dequantize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True` (default), perform image dequantization by (1/2, 1/2) translation in Fourier domain before computing the metric.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function. If `True`,
    the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – a method to reduce metric score over individual batch scores. `mean`: takes the mean, `sum` takes the sum, `none` or None no reduction will be applied (default).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalize images before passing to metric. `l2` normalizes by $\ell_2$ spatial norm, `min_max` normalizes by min and max of each input.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – If not `None` (default), center crop the tensor(s) before computing the metrics.
    If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
    If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
    If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

<hr />

* **Example:**

```pycon
>>> from deepinv.loss.metric import SharpnessIndex
>>> m = SharpnessIndex()
>>> x_net = torch.randn(2, 3, 16, 16)  # batch of 2 RGB images
>>> m(x_net).shape
torch.Size([2])
```

#### *static* dequant(u)

Image dequantization via (1/2, 1/2) translation in Fourier domain.

Adapted from MATLAB implementation in [https://helios2.mi.parisdescartes.fr/~moisan/sharpness/](https://helios2.mi.parisdescartes.fr/~moisan/sharpness/).

* **Parameters:**
  **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – (B, C, H, W) tensor
* **Returns:**
  (:class:torch.Tensor) dequantized image (B, C, H, W)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* logerfc(x)

Compute `log(erfc(x))` with asymptotic expansion for large `x`.

Adapted from MATLAB implementation in [https://helios2.mi.parisdescartes.fr/~moisan/sharpness/](https://helios2.mi.parisdescartes.fr/~moisan/sharpness/).

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(B, C, H, W)` tensor
* **Returns:**
  `(B,)` tensor of logarithmic value of `x`
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### metric(x_net, \*args, \*\*kwargs)

Compute sharpness index metric for a batch of images.

* **Parameters:**
  **x_net** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – (B, C, H, W) input tensors with C=1 or 3 channels.
* **Returns:**
  (B,) tensor of sharpness index values for each image in the batch
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* per_decomp(u)

Periodic + smooth decomposition of a 2D image.

Adapted from MATLAB implementation in [https://helios2.mi.parisdescartes.fr/~moisan/sharpness/](https://helios2.mi.parisdescartes.fr/~moisan/sharpness/).

* **Parameters:**
  **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – (B, C, H, W) tensor
* **Returns:**
  p: periodic component minus smooth component (B, C, H, W)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-loss-metric-sharpnessindex"></a>

## Examples using `SharpnessIndex`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div>
<!-- thumbnail-parent-div-close --></div>
