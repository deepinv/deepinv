# AngularSpectrumPropagation

### *class* deepinv.physics.AngularSpectrumPropagation(img_size, H, pixel_size=1.0, dtype=torch.cfloat, device='cpu', \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.md#deepinv.physics.LinearPhysics)

Linear angular-spectrum propagation with a user-defined transfer function.

This operator propagates a complex wave field $u$ according to

$$
P_H(u) = \mathcal{F}^{-1}\{H(f_x, f_y)\mathcal{F}\{u\}\}.
$$

The transfer function `H` can be supplied either as a tensor in the
unshifted ordering used by [`torch.fft.fft2()`](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft2.html#torch.fft.fft2), or as a callable with
signature `H(fx, fy)`. In the latter case, the callable is evaluated once
at construction on frequency grids expressed in cycles per unit of
`pixel_size`.

This class models propagation of an already formed complex field. Object
transmission functions such as $T(\delta, \beta)=\exp(g(\delta,
\beta))$ should be represented by a separate nonlinear
[`deepinv.physics.Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics) operator and composed with this one.

#### NOTE
The FFT implementation imposes periodic boundary conditions. Pad the
field and define `H` on the padded grid when circular wrap-around is
not appropriate for the experiment.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,*  *...* *]*) – Shape of an unbatched input. The final two
    entries are interpreted as `(height, width)`.
  * **H** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *Callable*) – Transfer-function tensor or callable
    `H(fx, fy)`. It must act pointwise without adding an output dimension. It may be (or return) a
    scalar, have shape `(height, width)`, or contain leading dimensions that
    broadcast to `img_size` (for example, `(channels, height, width)`).
  * **pixel_size** ([*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – Sampling pitch `dy, dx`. If `float`, then `dy, dy`, default 1.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – Complex dtype used to evaluate and store `H`.
    Default: `torch.cfloat`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – Device on which the frequency grid and
    transfer function are created. Default: `"cpu"`.

<hr />

* **Examples:**
  Define a quadratic phase transfer function directly:
  ```pycon
  >>> import torch
  >>> from deepinv.physics import AngularSpectrumPropagation
  >>> propagation = AngularSpectrumPropagation(
  ...     img_size=(1, 32, 48),
  ...     pixel_size=(2e-6, 3e-6),
  ...     H=lambda fx, fy: torch.exp(
  ...         -1j * 1e-10 * (fx.square() + fy.square())
  ...     ),
  ... )
  >>> x = torch.randn(2, 1, 32, 48, dtype=torch.cfloat)
  >>> propagation(x).shape
  torch.Size([2, 1, 32, 48])
  ```

#### A(x, \*\*kwargs)

Propagate a complex field without noise or sensor nonlinearities.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – complex input image, where the last two dimensions are `(height, width)`.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) `y`, the measurements, of the same shape as `x`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, \*\*kwargs)

Apply the Hermitian adjoint of the propagation operator.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – complex input image, where the last two dimensions are `(height, width)`.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) `x`, of the same shape as `y`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_dagger(y, rcond=None, \*\*kwargs)

Apply the Moore-Penrose pseudo-inverse.

The operator is diagonal in the Fourier domain, so the pseudo-inverse is closed-form, analogous to [`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.md#deepinv.physics.DecomposablePhysics).

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – complex input image, where the last two dimensions are `(height, width)`.
  * **rcond** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – frequencies whose magnitude `< rcond * max(abs(H))` are mapped to zero. If `None`, uses real component dtype machine precision.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) `x`, of the same shape as `y`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* frequency_grid(img_size, pixel_size=1.0, dtype=torch.float32, device='cpu')

Return unshifted spatial-frequency grids `(fx, fy)`.

Frequencies are in cycles per unit of `pixel_size` and follow the
ordering expected by [`torch.fft.fft2()`](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft2.html#torch.fft.fft2).

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,*  *...* *]*) – Shape whose final entries are
    `(height, width)`.
  * **pixel_size** ([*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – Scalar pitch or
    `(dy, dx)`.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – Real floating-point dtype for the grids.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – Grid device.
* **Returns:**
  The two grids `(fx, fy)`, each of shape `(height, width)`.
* **Return type:**
  [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

<a id="sphx-glr-backref-deepinv-physics-angularspectrumpropagation"></a>

## Examples using `AngularSpectrumPropagation`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs projected phase from preprocessed near-field holograms acquired at several propagation distances, see e.g. huhn2022Fast for details on the physics.">![](auto_examples/physics/images/thumb/sphx_glr_demo_fresnel_phase_retrieval_thumb.png)

[Multi-distance Fresnel phase retrieval](https://deepinv.org/auto_examples/physics/demo_fresnel_phase_retrieval.md)

  <div class="sphx-glr-thumbnail-title">Multi-distance Fresnel phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
