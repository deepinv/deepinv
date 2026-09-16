# DiffractionBlurGenerator3D

### *class* deepinv.physics.generator.DiffractionBlurGenerator3D(psf_size, zernike_index=tuple(range(4, 12)), fc=0.2, kb=0.25, max_zernike_amplitude=0.15, zernike_perturbation_amplitude=0.0, pupil_size=(512, 512), apodize=False, random_rotate=False, stepz_pixel=1.0, index_convention='noll', rng=None, device='cpu', dtype=torch.float32, \*\*kwargs)

Bases: [`PSFGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PSFGenerator.html.md#deepinv.physics.generator.PSFGenerator)

3D diffraction limited kernels using Zernike decomposition of the phase mask (Fresnel/Fraunhoffer diffraction theory).

This method simulates the propagation of the wavefront from the pupil plane
(frequency domain) to multiple defocus planes in the image space.
The pupil function is constructed using a Zernike polynomial decomposition
of the wavefront aberrations, see [`deepinv.physics.generator.DiffractionBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DiffractionBlurGenerator.html.md#deepinv.physics.generator.DiffractionBlurGenerator) for more details.

At each depth $z$, the pupil function is modulated by a phase term
corresponding to the axial wave vector $k_z$,
which is derived from the dispersion relation of light in free space.

$$
k_z = \sqrt{k_{\text{total}}^2 - k_{\text{lateral}}^2}
$$

where $k_{\text{total}}$ is the total wave number (`kb`) and $k_{\text{lateral}}$ is the lateral wave vector component.
The pupil function at depth $z$ is given by:

$$
P(x, y, z) = P(x, y, 0) \cdot \exp \left( - i 2 \pi \cdot k_z \cdot z \right)
$$

And the depth planes are sampled according to the `stepz_pixel` parameter, which defines the ratio between the physical size of the $z$ direction to that in the $x/y$ direction of the voxels in the 3D image.

The 3D PSF is then computed by square modulus of Fourier transform of the modulated pupil function at each depth plane, followed by normalization across the spatial dimensions.

#### NOTE
This class uses [`deepinv.physics.generator.DiffractionBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DiffractionBlurGenerator.html.md#deepinv.physics.generator.DiffractionBlurGenerator) under the hood to generate the pupil function at $z=0$. Refer to its documentation for more details.

* **Parameters:**
  * **psf_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – give in the order `(depth, height, width)` the size of the PSF to generate.
  * **zernike_index** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]* *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,*  *...* *]*) – 

    activated Zernike coefficients in the following `index_convention` convention.
    It can be either:
    > - a tuple of `int` corresponding to the Noll or ANSI indices, in which case the `index_convention` parameter is required to interpret them correctly.
    > - a tuple of `tuple[int, int]` corresponding to the standard radial-angular indexing $(n,m)$. In this case, the `index_convention` parameter is ignored.

    Defaults to `(4, 5, 6, 7, 8, 9, 10, 11)`, correspond to radial order `n` from 2 to 3 (included) and the spherical aberration.
    These correspond to the following aberrations: defocus, astigmatism, coma, trefoil and spherical aberration.
  * **fc** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – cutoff frequency `(NA/emission_wavelength) * pixel_size`. Should be in `[0, 1/4]` to respect Shannon, defaults to `0.2`.
  * **kb** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – wave number `(NI/emission_wavelength) * pixel_size` or `(NA/NI) * fc`. Must be greater than `fc`. Defaults to `0.3`.
  * **max_zernike_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum amplitude of Zernike coefficients. Defaults to 0.15.
  * **zernike_perturbation_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – amplitude of per-channel chromatic perturbations, defaults to `0`.
  * **pupil_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – pixel size to synthesize the super-resolved pupil. The higher the more precise, defaults to `(512, 512)`.
    If an `int` is given, a square pupil is considered.
  * **apodize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to apodize the PSF to reduce ringing effects. Defaults to `False`.
  * **random_rotate** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to randomly rotate the PSF in the xy plane. Defaults to `False`.
  * **stepz_pixel** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Ratio between the physical size of the $z$ direction to that in the $x/y$ direction of the voxels in the 3D image.
    Defaults to `1.0`.
  * **index_convention** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – convention for the Zernike indices, either `'noll'` (default) or `'ansi'`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator (default to `None`).
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device (default to `'cpu'`).
  * **dtype** ([*type*](https://docs.python.org/3.9/library/functions.html#type)) – data type (default to `torch.float32`).
  * **kwargs** – additional arguments for [`deepinv.physics.generator.DiffractionBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DiffractionBlurGenerator.html.md#deepinv.physics.generator.DiffractionBlurGenerator).

#### NOTE
- `NA`: numerical aperture,
- `NI`: refraction index of the immersion medium,
- `emission_wavelength`: wavelength of the light,
- `pixel_size`: physical size of the pixels in the $xy$ plane in the same unit as `emission_wavelength`.

<hr />

* **Examples:**

```pycon
>>> import torch
>>> from deepinv.physics.generator import DiffractionBlurGenerator3D
>>> generator = DiffractionBlurGenerator3D((21, 51, 51), stepz_pixel = 2, zernike_index=(3,), index_convention='ansi')
>>> print(generator.zernike_polynomials) # list of Zernike polynomials used
['Zernike(n = 2, m = -2) -- Oblique Astigmatism']
>>> dict = generator.step()
>>> filter = dict['filter']
>>> print(filter.shape)
torch.Size([1, 1, 21, 51, 51])
>>> batch_size = 2
>>> n_zernike = len(generator.generator2d.zernike_index)
>>> dict = generator.step(batch_size=batch_size, coeff=0.1 * torch.rand(batch_size, n_zernike))
>>> dict.keys()
dict_keys(['filter', 'pupil', 'coeff', 'fc'])
```

#### step(batch_size=1, coeff=None, angle=None, seed=None, fc=None, kb=None, max_zernike_amplitude=None, zernike_perturbation_amplitude=None, \*\*kwargs)

Generate a batch of PSF with a batch of Zernike coefficients

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of PSFs to generate.
  * **coeff** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – tensor containing the Zernike coefficients.
    If `None`, random coefficients are generated. Accepts `(B, K)` or `(B, C, K)`, exactly as in
    [`deepinv.physics.generator.DiffractionBlurGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.DiffractionBlurGenerator.html.md#deepinv.physics.generator.DiffractionBlurGenerator).
  * **angle** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(batch_size,)` angles in degrees for PSF rotation.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.
  * **fc** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – overrides `self.fc`
    for this call only. Accepts the same types as the constructor’s `fc`.
  * **kb** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – overrides `self.kb`
    for this call only. Accepts the same types as `fc`.
  * **max_zernike_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – overrides `self.max_zernike_amplitude` for this call only. Only used when `coeff` is `None`.
  * **zernike_perturbation_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – overrides `self.zernike_perturbation_amplitude` for this call only.
* **Returns:**
  dictionary with keys
  - `filter`: tensor of size `(B, C, depth, H, W)` batch of 3D PSFs,
  - `pupil`: the pupil function,
  - `coeff`: list of sampled Zernike coefficients in this realization,
  - `angle`: the random rotation angles in degrees if `random_rotate` is `True`, nothing otherwise.
  - `fc`: tensor of shape `(B, C)` with the cutoff frequencies used.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

#### *property* zernike_polynomials *: [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[str](https://docs.python.org/3.9/library/stdtypes.html#str)]*

List of Zernike polynomials used in the decomposition, with the corresponding aberration if available.

<a id="sphx-glr-backref-deepinv-physics-generator-diffractionblurgenerator3d"></a>

## Examples using `DiffractionBlurGenerator3D`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div>
<!-- thumbnail-parent-div-close --></div>
