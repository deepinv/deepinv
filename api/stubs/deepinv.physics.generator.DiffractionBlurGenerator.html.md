# DiffractionBlurGenerator

### *class* deepinv.physics.generator.DiffractionBlurGenerator(psf_size, zernike_index=tuple(range(4, 12)), fc=0.2, max_zernike_amplitude=0.15, zernike_perturbation_amplitude=0.0, pupil_size=(256, 256), apodize=False, random_rotate=False, center=False, index_convention='noll', device='cpu', dtype=torch.float32, rng=None)

Bases: [`PSFGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PSFGenerator.html.md#deepinv.physics.generator.PSFGenerator)

Diffraction limited blur generator.

Generates 2D diffraction PSFs in optics using Zernike decomposition of the phase mask (Fresnel/Fraunhoffer diffraction theory, Fourier optics).

Zernike polynomials are a sequence of orthogonal polynomials defined on the unit disk.
They are commonly used in optical systems to describe wavefront aberrations.

The PSF is modeled as:

$$
h(\cdot; \lambda) = \left| \mathcal{F} \left[ \mathbb{1}_{|\boldsymbol{\rho}| \leq 1} \cdot \exp \left( - i 2 \pi \sum_k \frac{a_k}{\lambda} z_k(\boldsymbol{\rho}) \right) \right](\cdot) \right|^2
$$

where $\boldsymbol{\rho}$ are normalized pupil-plane coordinates (on the unit disk),
$a_k$ are the Zernike coefficients **in physical units** (nm OPD, wavelength-independent),
$\lambda$ is the emission wavelength (nm), and $z_k$ are the Zernike polynomials.

The phase in waves is therefore $\theta_k(\lambda) = a_k / \lambda$, so the same
physical aberration produces a stronger wavefront error (in waves) at shorter wavelengths.

For multi-channel (multi-colour) imaging the generator supports a perturbation model:

$$
\theta_k^{(b,c)} = \underbrace{\theta_k^{(b)} \cdot \frac{\lambda_{\text{ref}}}{\lambda_c}}_{\text{monochromatic, rescaled}} + \underbrace{\Delta\theta_k^{(b,c)}}_{\text{chromatic perturbation}}
$$

where $\theta_k^{(b)}$ are base coefficients (in waves at $\lambda_{\text{ref}}$,
i.e. channel 0) shared across channels, and $\Delta\theta_k^{(b,c)}$ are small
per-channel perturbations (e.g. sample-induced dispersion). The cutoff frequency is also
wavelength-dependent:

$$
f_c^{(c)} = \frac{\mathrm{NA} \cdot p}{\lambda_c}
$$

where $\mathrm{NA}$ is the numerical aperture and $p$ is the pixel size.

See Lakshminarayanan and Fleck<sup>[1](#footcite-lakshminarayanan2011zernike)</sup>
[or this link](https://e-l.unifi.it/pluginfile.php/1055875/mod_resource/content/1/Appunti_2020_Lezione%2014_4_Zernikepolynomialsaguidefinal.pdf)
or [`deepinv.physics.generator.Zernike`](https://deepinv.org/api/stubs/deepinv.physics.generator.Zernike.html.md#deepinv.physics.generator.Zernike) for more details.

In the ideal diffraction-limited case (i.e., no aberrations), the PSF corresponds to the Airy pattern.

The Zernike polynomials $z_k$ are indexed using the `'noll'` or `'ansi'` convention (defined by `index_convention` parameter).
Conversion from the two conventions to the standard radial-angular indexing is done internally (see [wikipedia page](https://en.wikipedia.org/wiki/Zernike_polynomials)).

* **Parameters:**
  * **psf_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – the shape `H x W` of the generated PSF in 2D
  * **zernike_index** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]* *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,*  *...* *]*) – 

    activated Zernike coefficients in the following `index_convention` convention.
    It can be either:
    > - a tuple of `int` corresponding to the Noll or ANSI indices, in which case the `index_convention` parameter is required to interpret them correctly.
    > - a tuple of `tuple[int, int]` corresponding to the standard radial-angular indexing $(n,m)$. In this case, the `index_convention` parameter is ignored.

    Defaults to `(4, 5, 6, 7, 8, 9, 10, 11)`, correspond to radial order `n` from 2 to 3 (included) and the spherical aberration.
    These correspond to the following aberrations: defocus, astigmatism, coma, trefoil and spherical aberration.
  * **fc** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

    default cutoff frequency
    `(NA/emission_wavelength) * pixel_size`. Should be in `[0, 0.25]` to respect the
    Shannon-Nyquist sampling theorem, defaults to `0.2`.

    At **construction time**, only a scalar `float` or a 1D tensor/sequence of length `C`
    are accepted. A 2D tensor raises a `ValueError`.

    At **step time** (passed to [`step()`](#deepinv.physics.generator.DiffractionBlurGenerator.step)), `fc` may additionally be a 2D tensor of
    shape `(B, C)` for full per-(batch, channel) control. The output PSF shape is then
    driven by `fc` as follows:
    > - `float` / scalar: `(batch_size, 1, H, W)`.
    > - `(C,)` 1D tensor/sequence: `(batch_size, C, H, W)`.
    > - `(B, C)` 2D tensor: `(B, C, H, W)`.
    > - `(1, C)` 2D tensor: `(batch_size, C, H, W)`.
    > - `(B, 1)` 2D tensor: `(B, 1, H, W)`.
  * **max_zernike_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – default amplitude of the base Zernike coefficients (in
    waves at the channel-0/reference cutoff frequency), defaults to `0.15`. The amplitude
    of each coefficient is sampled uniformly in `[-max_zernike_amplitude/2, max_zernike_amplitude/2]`.
    Can be overridden per [`step()`](#deepinv.physics.generator.DiffractionBlurGenerator.step) call.
  * **zernike_perturbation_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – default amplitude of the per-channel chromatic
    perturbations, defaults to `0`. Can be overridden per [`step()`](#deepinv.physics.generator.DiffractionBlurGenerator.step) call.
  * **pupil_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – pixel size used to synthesize the super-resolved pupil.
    The higher the more precise, defaults to `(256, 256)`.
    If a single `int` is given, a square pupil is considered.
  * **apodize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to apodize the PSF to reduce ringing artifacts, defaults to `False`.
  * **random_rotate** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to randomly rotate the PSF, defaults to `False`.
  * **center** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to center the barycenter of the PSF, defaults to `False`. Less effective if either `random_rotate` or `apodize` is True
  * **index_convention** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the convention for the Zernike polynomials indexing. Can be either `'noll'` (default) or `'ansi'`.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device where the tensors are allocated and processed, defaults to `'cpu'`.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the tensors, defaults to `torch.float32`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – pseudo random number generator for reproducibility. Defaults to `None`.

<hr />

* **Examples:**

```pycon
>>> from deepinv.physics.generator import DiffractionBlurGenerator
>>> generator = DiffractionBlurGenerator((5, 5))
>>> print("\n".join(generator.zernike_polynomials))
Zernike(n = 2, m = 0) -- Defocus
Zernike(n = 2, m = -2) -- Oblique Astigmatism
Zernike(n = 2, m = 2) -- Vertical Astigmatism
Zernike(n = 3, m = -1) -- Vertical Coma
Zernike(n = 3, m = 1) -- Horizontal Coma
Zernike(n = 3, m = -3) -- Vertical Trefoil
Zernike(n = 3, m = 3) -- Oblique Trefoil
Zernike(n = 4, m = 0) -- Primary Spherical
>>> blur = generator.step()  # dict_keys(['filter', 'coeff', 'pupil'])
>>> print(blur['filter'].shape)
torch.Size([1, 1, 5, 5])
```

```pycon
>>> generator = DiffractionBlurGenerator((5, 5), fc=(0.18, 0.20, 0.22))
>>> blur = generator.step(batch_size=2)
>>> print(blur['filter'].shape)
torch.Size([2, 3, 5, 5])
>>> print(blur['coeff'].shape)   # (B, C, K): wavelength-rescaled base + chromatic perturbations
torch.Size([2, 3, 8])
```

<hr />

* **References:**

* <a id='footcite-lakshminarayanan2011zernike'>**[1]**</a> Vasudevan Lakshminarayanan and Andre Fleck. Zernike polynomials: a guide. *Journal of Modern Optics*, 58(7):545–561, 2011.

#### generate_angles(batch_size)

Generate random rotation angles for the PSF.

* **Parameters:**
  **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batch_size.
* **Returns:**
  `(batch_size,)` angles in degrees.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### generate_coeff(batch_size, fc=None, max_zernike_amplitude=None, zernike_perturbation_amplitude=None, n_zernike=None)

Generate random Zernike coefficients, scaled by cutoff frequency per channel.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of independent aberration realizations.
  * **fc** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – already-formatted `(B, C)` tensor from
    class method `_format_fc()`. If `None`, `self.fc` is used with `batch_size`,
    producing a `(batch_size, K)` output (backward-compatible behaviour).
  * **max_zernike_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – amplitude of the base coefficients.
    Defaults to `self.max_zernike_amplitude`.
  * **zernike_perturbation_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – amplitude of per-channel
    perturbations. Defaults to `self.zernike_perturbation_amplitude`.
  * **n_zernike** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of Zernike coefficients to generate. Defaults to
    `self.n_zernike`. Set to `len(used_zernike_index)` when called from
    [`step()`](#deepinv.physics.generator.DiffractionBlurGenerator.step) with a `used_zernike_index` argument.
* **Returns:**
  `(batch_size, K)` if `C == 1`, else `(B, C, K)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### step(batch_size=1, coeff=None, angle=None, center=None, max_zernike_amplitude=None, zernike_perturbation_amplitude=None, seed=None, fc=None, used_zernike_index=None, \*\*kwargs)

Generate a batch of PSFs with a batch of Zernike coefficients.

The shape of the output PSF is determined by `fc` as follows:

> - `None`: `(batch_size, 1, *self.psf_size)` or `(batch_size, len(self.fc), *self.psf_size)`.
> - `float` / scalar: `(batch_size, 1, *self.psf_size)`.
> - `(C,)` 1D tensor/sequence: `(batch_size, C, *self.psf_size)`.
> - `(B, C)` 2D tensor: `(B, C, *self.psf_size)`.
* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of PSFs to generate. Ignored when `fc` is a 2D
    tensor with `B > 1` (batch size is then read from `fc`). Defaults to `1`.
  * **coeff** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

    Zernike coefficients. Accepted shapes:
    - `None` (default): sampled via [`generate_coeff()`](#deepinv.physics.generator.DiffractionBlurGenerator.generate_coeff).
    - `(B, n_zernike_used)`: base coefficients per batch element, shared across
      channels. No rescaling applied. No chromatic perturbation is added.
    - `(B, C, n_zernike_used)`: fully specified per channel. No rescaling applied.
  * **angle** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – `(batch_size,)` angles in degrees for PSF rotation.
  * **max_zernike_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – overrides `self.max_zernike_amplitude`
    for this call only. Only used when `coeff` is `None`.
  * **zernike_perturbation_amplitude** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – overrides
    `self.zernike_perturbation_amplitude` for this call only.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.
  * **fc** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]* *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – overrides `self.fc`
    for this call only (does not mutate `self.fc`). Defaults to `None`, in which
    case `self.fc` is used. See class docstring for accepted shapes and their effect
    on the output PSF shape.
  * **used_zernike_index** – 

    subset of Zernike indices to activate for this call.
    Must be a sub-sequence of `self.zernike_index` (same format: ints or `(n, m)`
    tuples). `None` (default) uses the full set `self.zernike_index`. Useful for
    varying the active polynomial set without re-instantiating the generator:
    ```default
    gen = DiffractionBlurGenerator((31, 31), zernike_index=range(3, 37))
    p1 = gen.step(used_zernike_index=range(3, 16))
    p2 = gen.step(used_zernike_index=range(3, 28))
    ```
* **Returns:**
  dictionary with keys
  - `filter`: tensor of size `(B, C, H, W)` where `B` and `C` are
    determined by `fc` as described above,
  - `coeff`: the Zernike coefficients actually used, shape
    `(B, n_zernike_used)` or `(B, C, n_zernike_used)` where
    `n_zernike_used = len(used_zernike_index)` if specified, else
    `self.n_zernike`,
  - `pupil`: the pupil function,
  - `fc`: tensor of shape `(Bf, Cf)` with the cutoff frequencies actually used,
  - `angle`: the random rotation angle in degrees if `random_rotate` is `True`,
  - `coeff_tilt_x`: the Zernike coefficient for tilt in the x-direction if center is `True`,
  - `coeff_tilt_y`: the Zernike coefficient for tilt in the y-direction if center is `True`.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

#### *property* zernike_polynomials *: [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[str](https://docs.python.org/3.9/library/stdtypes.html#str)]*

List of Zernike polynomials used in the decomposition, with the corresponding aberration if available.

<a id="sphx-glr-backref-deepinv-physics-generator-diffractionblurgenerator"></a>

## Examples using `DiffractionBlurGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
