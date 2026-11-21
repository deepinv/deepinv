# FresnelPropagation

### *class* deepinv.physics.FresnelPropagation(img_size, wavelength, distance, pixel_size, dtype=torch.cfloat, device='cpu', \*\*kwargs)

Bases: [`AngularSpectrumPropagation`](https://deepinv.org/api/stubs/deepinv.physics.AngularSpectrumPropagation.md#deepinv.physics.AngularSpectrumPropagation)

Fresnel (paraxial angular-spectrum) propagation of a complex wave field.

For wavelength $\lambda$, propagation distance $z$, and spatial
frequencies $(f_x, f_y)$, this class uses

$$
H(f_x, f_y) = \exp\left[-\mathrm{i}\pi\lambda z
(f_x^2 + f_y^2)\right].
$$

This transfer function assumes a coherent plane wave illumination and paraxial approximation.
It is used in the near-field case where the propagation distance $z \ll a^2/\lambda$,
with $a$ being the feature size of the object (or pixel size).

#### NOTE
To model in-line holography, compose this physics with [`deepinv.physics.PhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.md#deepinv.physics.PhaseRetrieval) and a transmission function.
To model propagation-based phase-contrast CT, additionally compose with [`deepinv.physics.Tomography`](https://deepinv.org/api/stubs/deepinv.physics.Tomography.md#deepinv.physics.Tomography) or [`deepinv.physics.TomographyWithAstra`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.md#deepinv.physics.TomographyWithAstra).

#### HINT
Holography geometries are often described in terms of the Fresnel number.
The Fresnel number is defined from the physical parameter as

$$
a_F = \frac{a^2}{\lambda z},

$$

with $a$ being the characteristic size of the object, typically the pixel size, $\lambda$ the wavelength, and $z$ the propagation distance.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,*  *...* *]*) – Shape of an unbatched input. The final two
    entries are `(height, width)`.
  * **wavelength** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Wavelength in the same length unit as `distance`
    and `pixel_size`.
  * **distance** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Propagation distance.
  * **pixel_size** ([*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – Sampling pitch `dy, dx`. If `float`, then `dy, dy`, default 1.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – Complex dtype of the transfer function. Default:
    `torch.cfloat`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – Device on which the operator is created.

<hr />

* **Examples:**
  Using FresnelPropagation for phase retrieval
  ```pycon
  >>> import torch
  >>> from deepinv.physics import FresnelPropagation, PhaseRetrieval
  >>> B = FresnelPropagation(
  ...     img_size=(1, 64, 64),
  ...     wavelength=500e-9,
  ...     distance=0.1,
  ...     pixel_size=5e-6,
  ... )
  >>> physics = PhaseRetrieval(B=B)
  >>> transmission = torch.randn(2, 1, 64, 64, dtype=torch.cfloat)
  >>> physics(transmission).dtype
  torch.float32
  >>> physics(transmission).size()
  torch.Size([2, 1, 64, 64])
  ```

  If a transmission function should be added, proceed like:
  ```pycon
  >>> from deepinv.physics import Physics, compose
  >>> k = 2 * torch.pi / 500e-9
  >>> transmission_model = Physics(
  ...     A=lambda x: torch.exp(-1j * k * x[:, 0:1] - k * x[:, 1:2])
  ... )
  >>> physics = compose(transmission_model, physics)
  ```

<a id="sphx-glr-backref-deepinv-physics-fresnelpropagation"></a>

## Examples using `FresnelPropagation`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs projected phase from preprocessed near-field holograms acquired at several propagation distances, see e.g. huhn2022Fast for details on the physics.">![](auto_examples/physics/images/thumb/sphx_glr_demo_fresnel_phase_retrieval_thumb.png)

[Multi-distance Fresnel phase retrieval](https://deepinv.org/auto_examples/physics/demo_fresnel_phase_retrieval.md)

  <div class="sphx-glr-thumbnail-title">Multi-distance Fresnel phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
