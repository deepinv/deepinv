# ProductConvolutionBlurGenerator

### *class* deepinv.physics.generator.ProductConvolutionBlurGenerator(psf_generator, img_size, n_eigen_psf=10, spacing=None, device='cpu', \*\*kwargs)

Bases: [`PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)

Generates parameters of space-varying blurs.

Parameters generated:

-`'filters'`: tensor of shape `(B, C, n_eigen_psf, psf_size, psf_size)`
- ‘multipliers’: tensor of shape `(B, C, n_eigen_psf, H, W)`

See [`deepinv.physics.SpaceVaryingBlur`](https://deepinv.org/api/stubs/deepinv.physics.SpaceVaryingBlur.html.md#deepinv.physics.SpaceVaryingBlur) for more details.

* **Parameters:**
  * **psf_generator** ([*deepinv.physics.generator.PSFGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PSFGenerator.html.md#deepinv.physics.generator.PSFGenerator)) – A PSF generator, such as [`motion blur`](https://deepinv.org/api/stubs/deepinv.physics.generator.MotionBlurGenerator.html.md#deepinv.physics.generator.MotionBlurGenerator) or
    [`diffraction blur generator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DiffractionBlurGenerator.html.md#deepinv.physics.generator.DiffractionBlurGenerator).
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – image size `(H,W)`.
  * **n_eigen_psf** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – each PSF in the field of view will be a linear combination of `n_eigen_psf` eigen PSF grids.
    Defaults to 10.
  * **spacing** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – steps between the PSF grids used for interpolation (defaults `(H//8, W//8)`).
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – boundary conditions in (options = `'valid'`, `'circular'`, `'replicate'`, `'reflect'`).
    Defaults to `'valid'`.

<hr />

* **Examples:**

```pycon
>>> from deepinv.physics.generator import DiffractionBlurGenerator
>>> from deepinv.physics.generator import ProductConvolutionBlurGenerator
>>> psf_size = 7
>>> psf_generator = DiffractionBlurGenerator((psf_size, psf_size), fc=0.25)
>>> pc_generator = ProductConvolutionBlurGenerator(psf_generator, img_size=(64, 64), n_eigen_psf=8)
>>> params = pc_generator.step(1)
>>> print(params.keys())
dict_keys(['filters', 'multipliers'])
```

#### step(batch_size=1, seed=None, \*\*kwargs)

Generates a random set of filters and multipliers for space-varying blurs.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of space-varying blur parameters to generate.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.
* **Returns:**
  a dictionary containing filters, multipliers and paddings.
  filters: a tensor of shape (B, C, n_eigen_psf, psf_size, psf_size).
  multipliers: a tensor of shape (B, C, n_eigen_psf, H, W).

<a id="sphx-glr-backref-deepinv-physics-generator-productconvolutionblurgenerator"></a>

## Examples using `ProductConvolutionBlurGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
