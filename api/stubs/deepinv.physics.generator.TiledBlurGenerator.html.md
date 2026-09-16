# TiledBlurGenerator

### *class* deepinv.physics.generator.TiledBlurGenerator(psf_generator, patch_size, stride=None, rng=None, device='cpu', \*\*kwargs)

Bases: [`TiledMixin2d`](https://deepinv.org/api/stubs/deepinv.utils.TiledMixin2d.html.md#deepinv.utils.TiledMixin2d), [`PSFGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PSFGenerator.html.md#deepinv.physics.generator.PSFGenerator)

Generates parameters of the [`deepinv.physics.TiledSpaceVaryingBlur`](https://deepinv.org/api/stubs/deepinv.physics.TiledSpaceVaryingBlur.html.md#deepinv.physics.TiledSpaceVaryingBlur) operator.
The image is divided into overlapping patches, each local patch is convolved with a different PSF.

This generates a dict with key `'filter'`, which is tensor of shape `(B, C, K, psf_size, psf_size)`
where `K` is the number of patches in which the image is divided.
It is computed based on the `patch_size`, `stride` and the given `img_size` during the `step()` function call.

* **Parameters:**
  * **psf_generator** ([*deepinv.physics.generator.PSFGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PSFGenerator.html.md#deepinv.physics.generator.PSFGenerator)) – A PSF generator, such as [`motion blur`](https://deepinv.org/api/stubs/deepinv.physics.generator.MotionBlurGenerator.html.md#deepinv.physics.generator.MotionBlurGenerator) or [`diffraction blur generator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DiffractionBlurGenerator.html.md#deepinv.physics.generator.DiffractionBlurGenerator).
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the patches (height, width) in which the image is divided.
  * **stride** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – stride between adjacent patches (height, width). Defaults to `patch_size`.

#### step(batch_size=1, img_size=None, seed=None, \*\*kwargs)

Generates a random set of filters for the tiled space-varying blur.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batch size of the PSF parameters to generate. Should be equal to the batch size of the images to be blurred.
  * **img_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the image to be blurred (height, width).
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* *None*) – the seed for the random number generator.
* **Returns:**
  a dictionary containing filters, with key:
  - `filters`: a tensor of shape `(B, C, K, psf_size, psf_size)`, where `K` is the number of patches in which the image is divided.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

<a id="sphx-glr-backref-deepinv-physics-generator-tiledblurgenerator"></a>

## Examples using `TiledBlurGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
