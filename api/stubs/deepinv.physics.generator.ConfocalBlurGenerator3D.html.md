# ConfocalBlurGenerator3D

### *class* deepinv.physics.generator.ConfocalBlurGenerator3D(psf_size, zernike_index=tuple(range(4, 12)), NI=1.51, NA=1.37, lambda_ill=489e-9, lambda_coll=395e-9, pixelsize_XY=50e-9, pixelsize_Z=100e-9, pinhole_radius=1, max_zernike_amplitude=0.1, zernike_perturbation_amplitude=0.0, pupil_size=(512, 512), index_convention='noll', device='cpu', dtype=torch.float32, rng=None, \*\*kwargs)

Bases: [`PSFGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PSFGenerator.html.md#deepinv.physics.generator.PSFGenerator)

Generates the 3D point spread function (PSF) of a confocal laser scanning microscope.

* **Parameters:**
  * **psf_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – give in the order `(depth, height, width)`
  * **zernike_index** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]* *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,*  *...* *]*) – 

    activated Zernike coefficients in the following `index_convention` convention.
    It can be either:
    > - a tuple of `int` corresponding to the Noll or ANSI indices, in which case the `index_convention` parameter is required to interpret them correctly.
    > - a tuple of `tuple[int, int]` corresponding to the standard radial-angular indexing $(n,m)$. In this case, the `index_convention` parameter is ignored.

    Defaults to `(4, 5, 6, 7, 8, 9, 10, 11)`, correspond to radial order `n` from 2 to 3 (included) and the spherical aberration.
    These correspond to the following aberrations: defocus, astigmatism, coma, trefoil and spherical aberration.
  * **NI** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Refractive index of  the immersion medium. Defaults to `1.51` (oil),
  * **NA** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Numerical aperture. Should be less than NI. Defaults to `1.37`.
  * **lambda_ill** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – Wavelength(s) of the illumination light (fluorescence excitation). Defaults to `489e-9`.
    Pass a list of `C` values to generate multi-colour PSFs (one channel per wavelength).
  * **lambda_coll** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – Wavelength(s) of the collection light (fluorescence emission). Defaults to `395e-9`.
    Must have the same length as `lambda_ill` when a list is provided.
  * **pixelsize_XY** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Physical pixel size in the lateral direction (height, width). Defaults to `50e-9`.
  * **pixelsize_Z** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Physical pixel size in the axial direction (depth). Defaults to `100e-9`.
  * **pinhole_radius** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Radius of pinhole in Airy units. Defaults to `1`.
  * **max_zernike_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum amplitude of Zernike coefficients. Defaults to `0.1`.
  * **zernike_perturbation_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – amplitude of per-channel chromatic perturbations, defaults to `0`.
  * **pupil_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – pixel size to synthesize the super-resolved pupil. The higher the more precise, defaults to `(512, 512)`.
    If an `int` is given, a square pupil is considered.
  * **index_convention** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – convention for the Zernike indices, either `'noll'` (default) or `'ansi'`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator (default to `None`).
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device (default to `cpu`).
  * **dtype** ([*type*](https://docs.python.org/3.9/library/functions.html#type)) – data type (default to `torch.float32`).

<hr />

* **Examples:**

```pycon
>>> import torch
>>> from deepinv.physics.generator import ConfocalBlurGenerator3D
>>> generator = ConfocalBlurGenerator3D((21, 51, 51), zernike_index=(3,))
>>> print(generator.zernike_polynomials)
['Zernike(n = 1, m = -1) -- Vertical Tilt']
>>> dict = generator.step()
>>> filter = dict['filter']
>>> print(filter.shape)
torch.Size([1, 1, 21, 51, 51])
>>> batch_size = 2
>>> n_zernike = len(generator.generator_ill.generator2d.zernike_index)
>>> dict = generator.step(batch_size=batch_size,
...                       coeff_ill = 0.1 * torch.rand(batch_size, n_zernike),
...                       coeff_coll = 0.1 * torch.rand(batch_size, n_zernike))
>>> dict.keys()
dict_keys(['filter', 'pupil_ill', 'pupil_coll', 'coeff_ill', 'coeff_coll', 'fc_ill', 'fc_coll'])
```

Multi-colour example (one channel per excitation/emission wavelength pair):

```pycon
>>> generator = ConfocalBlurGenerator3D(
...     (21, 51, 51),
...     lambda_ill=[489e-9, 561e-9],
...     lambda_coll=[525e-9, 620e-9],
...     zernike_index=(3,),
... )
>>> dict = generator.step()
>>> print(dict['filter'].shape)
torch.Size([1, 2, 21, 51, 51])
```

#### step(batch_size=1, seed=None, coeff_ill=None, coeff_coll=None, fc_ill=None, kb_ill=None, fc_coll=None, kb_coll=None, \*\*kwargs)

Generate a batch of 3D confocal PSF with a batch of Zernike coefficients
for illumination and collection

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of PSFs to generate.
  * **coeff_ill** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – tensor of size `batch_size x len(zernike_index)` containing the Zernike coefficients for illumination.
    If `None`, random coefficients are generated.
  * **coeff_coll** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – tensor of size `batch_size x len(zernike_index)` containing the Zernike coefficients for collection.
    If `None`, random coefficients are generated.
  * **fc_ill** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – overrides the illumination cutoff frequency for this call only.
  * **kb_ill** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – overrides the illumination wave number for this call only.
  * **fc_coll** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – overrides the collection cutoff frequency for this call only.
  * **kb_coll** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – overrides the collection wave number for this call only.
* **Returns:**
  dictionary with keys
  - `filter`: tensor of size `batch_size x C x psf_size[0] x psf_size[1] x psf_size[2]` batch of PSFs,
  - `coeff_ill`: list of sampled Zernike coefficients in this realization of illumination,
  - `coeff_coll`: list of sampled Zernike coefficients in this realization of collection,
  - `pupil_ill`: the illumination pupil function,
  - `pupil_coll`: the collection pupil function,
  - `fc_ill`: tensor of shape `(B, C)` with the illumination cutoff frequencies used,
  - `fc_coll`: tensor of shape `(B, C)` with the collection cutoff frequencies used.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

#### *property* zernike_polynomials *: [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[str](https://docs.python.org/3.9/library/stdtypes.html#str)]*

List of Zernike polynomials used in the decomposition, with the corresponding aberration if available.

<a id="sphx-glr-backref-deepinv-physics-generator-confocalblurgenerator3d"></a>

## Examples using `ConfocalBlurGenerator3D`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div>
<!-- thumbnail-parent-div-close --></div>
