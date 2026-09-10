# TiledSpaceVaryingBlur

### *class* deepinv.physics.TiledSpaceVaryingBlur(filters=None, patch_size=None, stride=None, blending_mode='bump', use_fft=True, device='cpu', dtype=torch.float32, \*\*kwargs)

Bases: [`TiledMixin2d`](https://deepinv.org/api/stubs/deepinv.utils.TiledMixin2d.html.md#deepinv.utils.TiledMixin2d), [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Space varying blur via tiled-convolution.

This forward operator performs the space-varying blur using local convolutions on overlapping patches (tiles) of the image.
The resulting images is then reconstructed using a smooth blending of the overlapping patches (`blending_mode`).

Given an input image $x$, this linear operator performs

$$
y = \sum_{k=1}^K h_k \star (m_k \odot x)
$$

where $\star$ is a convolution, $\odot$ is a Hadamard product, $m_k$ are binary masks defining the tiles and $h_k$ are filters.
When the size of the image minus `stride` is not perfectly divisible by the `patch_size`, the image is padded with zeros to fit an integer number of patches and the extra padding is removed afterwards automatically.

The number of patches (tiles) $K$ is determined by the `patch_size` and `stride` parameters: it is the product of the number of patches along each spatial dimension.
A helper class method `num_filters(img_size, patch_size, stride)` is provided to compute the number of filters needed for a given image size, patch size and stride.

#### NOTE
For simplicity, we also provide the generator class [`deepinv.physics.generator.TiledBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.TiledBlurGenerator.html.md#deepinv.physics.generator.TiledBlurGenerator) to generate the filters, for example during training.

#### NOTE
This class supports both FFT-based convolutions and direct convolutions, see [`deepinv.physics.functional.conv2d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d_fft.html.md#deepinv.physics.functional.conv2d_fft) and [`deepinv.physics.functional.conv2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d.html.md#deepinv.physics.functional.conv2d).
FFT-based convolutions are typically faster for large filters. This can be selected via the `use_fft` parameter (default: `True`).

#### NOTE
This class supports broadcast between of batch and channel dimensions of the filters and the input image.
See [`deepinv.physics.functional.conv2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d.html.md#deepinv.physics.functional.conv2d) for more details.

* **Parameters:**
  * **filters** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filters $h_k$. Tensor of size `(B, C, K, h, w)` where
    `B` is the batch size, `C` the number of channels, `K` the number of filters, `h` and `w` the filter height and width which should be smaller or equal than the image $x$ height and width respectively.
    If `None`, filters must be provided during the forward pass.
  * **blending_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Blending mode for overlapping patches. Options are `'bump'` (default) and `'linear'`.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Size of the patches (tiles). If `int`, the same size is used for both spatial dimensions.

<hr />

#### NOTE
This class only supports `'valid'` padding. If you need other padding options, please raise an issue.

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
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Image size `(H, W)`.
  * **patch_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Patch size `(h, w)`.
  * **stride** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Stride size `(sh, sw)`. If `None`, it is set equal to `patch_size // 2` (default).
* **Returns:**
  Number of filters (tiles) required in each spatial dimension.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[int](https://docs.python.org/3.9/library/functions.html#int), [int](https://docs.python.org/3.9/library/functions.html#int)]

<a id="sphx-glr-backref-deepinv-physics-tiledspacevaryingblur"></a>

## Examples using `TiledSpaceVaryingBlur`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
