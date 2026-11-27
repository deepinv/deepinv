# TiledSpaceVaryingBlur

### *class* deepinv.physics.TiledSpaceVaryingBlur(filters=None, patch_size=None, stride=None, blending_mode='bump', use_fft=True, device='cpu', dtype=torch.float32, mask_first=False, \*\*kwargs)

Bases: [`TiledMixin2d`](https://deepinv.org/api/stubs/deepinv.utils.TiledMixin2d.md#deepinv.utils.TiledMixin2d), [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.md#deepinv.physics.LinearPhysics)

Space-varying blur using local convolutions on overlapping tiles.

This forward operator performs the space-varying blur using local convolutions on overlapping patches (tiles) of the image.
The resulting image is then reconstructed using a smooth blending of the overlapping patches (`blending_mode`).

Given an input image $x$, with `mask_first=False` (default), this linear operator performs

$$
y = \sum_{k=1}^K w_k \odot (h_k \star x).
$$

whereas if `mask_first=True`, the multipliers are applied before the convolutions:

$$
y = \sum_{k=1}^K h_k \star (w_k \odot x).
$$

where $\star$ is a convolution, $\odot$ is a Hadamard product, $w_k$ are smooth masks defining the tiles and forming a partition of unity, and $h_k$ are filters.
When the image size is not compatible with `patch_size` and `stride`, the image is padded with zeros to fit an integer number of patches and the extra padding is removed afterwards automatically.

The number of patches (tiles) $K$ is determined by the `patch_size` and `stride` parameters: it is the product of the number of patches along each spatial dimension.
A helper class method `num_filters(img_size, patch_size, stride)` is provided to compute the number of filters needed for a given image size, patch size and stride.

#### NOTE
For simplicity, we also provide the generator class [`deepinv.physics.generator.TiledBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.TiledBlurGenerator.md#deepinv.physics.generator.TiledBlurGenerator) to generate the filters, for example during training.

#### NOTE
This class supports both FFT-based convolutions and direct convolutions, see [`deepinv.physics.functional.conv2d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d_fft.md#deepinv.physics.functional.conv2d_fft) and [`deepinv.physics.functional.conv2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d.md#deepinv.physics.functional.conv2d).
FFT-based convolutions are typically faster for large filters. This can be selected via the `use_fft` parameter (default: `True`).

#### NOTE
This class supports broadcasting between the batch and channel dimensions of the filters and the input image.
See [`deepinv.physics.functional.conv2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d.md#deepinv.physics.functional.conv2d) for more details.

* **Parameters:**
  * **filters** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – Filters $h_k$. Tensor of size `(B, C, K, h, w)` where
    `B` is the batch size, `C` the number of channels, `K` the number of filters, `h` and `w` the filter height and width which should be smaller or equal than the image $x$ height and width respectively.
    If `None`, filters must be provided during the forward pass.
  * **blending_mode** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – Blending mode for overlapping patches. Options are `'bump'` (default) and `'linear'`.
  * **patch_size** ([*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – Size of the patches (tiles). If `int`, the same size is used for both spatial dimensions.
  * **stride** ([*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]* *,* *None*) – Stride between patches. Defaults to half the patch size.
  * **use_fft** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – Use FFT-based rather than direct convolutions. Default: `True`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – Device for the filters and blending weights.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – Data type for the filters and blending weights.
  * **mask_first** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – Apply the weights before (`True`) or after (`False`, default) convolution. Use `True` for filters indexed by source position, such as PSFs measured from point sources, and `False` for filters indexed by output position. Unlike [`SpaceVaryingBlur`](https://deepinv.org/api/stubs/deepinv.physics.SpaceVaryingBlur.md#deepinv.physics.SpaceVaryingBlur), this class defaults to `False`.

<hr />

#### NOTE
This class only supports `'valid'` padding. An input of size `(H, W)` and filters of size `(h, w)` produce an output of size `(H-h+1, W-w+1)` in both modes.
With `mask_first=True`, full convolution outputs are accumulated across tile boundaries before the global valid crop.

<hr />

* **Examples:**

```pycon
>>> import torch
>>> from deepinv.physics import TiledSpaceVaryingBlur
>>> from deepinv.physics.generator import MotionBlurGenerator, TiledBlurGenerator
>>> import deepinv as dinv
>>> img_size = (256, 256)
>>> patch_size = (64, 64)
>>> stride = (32, 32)
>>> x = dinv.utils.load_example(
...        "butterfly.png", img_size=img_size, resize_mode="resize"
...    )
>>> psf_generator = MotionBlurGenerator(psf_size=(31, 31))
>>> generator = TiledBlurGenerator(
...     psf_generator=psf_generator,
...     patch_size=patch_size,
...     stride=stride,
... )
>>> filters = generator.step(batch_size=1, img_size=img_size)["filters"]
>>> physics = TiledSpaceVaryingBlur(patch_size=patch_size, stride=stride)
>>> y = physics(x, filters=filters)
>>> print(x.shape, y.shape)
torch.Size([1, 3, 256, 256]) torch.Size([1, 3, 226, 226])
>>> dinv.utils.plot([x, y], titles=["Original", "Blurred"])
```

#### A(x, filters=None, \*\*kwargs)

Applies the space varying blur operator to the input image.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
  * **filters** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filters $h_k$.
* **Returns:**
  torch.Tensor: Space varying blurred image.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, filters=None, \*\*kwargs)

Applies the adjoint operator.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – blurry image $y$. Tensor of size `(B, C, H', W')`.
  * **filters** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filters $h_k$. Tensor of size `(b, c, K, h, w)`.
* **Returns:**
  torch.Tensor: Adjoint result.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* num_filters(img_size, patch_size, stride=None)

Computes the number of filters (tiles) required for a given image size, patch size and stride.
Can be used to determine the required number of filters when instantiating the class.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – Image size `(H, W)`.
  * **patch_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – Patch size `(h, w)`.
  * **stride** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]*) – Stride size `(sh, sw)`. If `None`, it is set equal to `patch_size // 2` (default).
* **Returns:**
  Number of filters (tiles) required in each spatial dimension.
* **Return type:**
  [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[int](https://docs.python.org/3.12/library/functions.html#int), [int](https://docs.python.org/3.12/library/functions.html#int)]

<a id="sphx-glr-backref-deepinv-physics-tiledspacevaryingblur"></a>

## Examples using `TiledSpaceVaryingBlur`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">![](auto_examples/physics/images/thumb/sphx_glr_demo_blur_tour_thumb.png)

[Tour of blur operators](https://deepinv.org/auto_examples/physics/demo_blur_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
