# UltrasoundPlaneWave

### *class* deepinv.physics.UltrasoundPlaneWave(img_size, angles, element_positions, n_samples, sampling_frequency, sound_speed=1540.0, t0=0.0, , pixel_grid=None, pixel_size=None, pixel_origin=None, f_number=None, receive_apod_window='rect', transmit_apod_window=None, pulse=None, normalize=False, device='cpu')

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

2D plane-wave ultrafast ultrasound imaging operator.

Models the linear operator $A$ mapping an image $x$ to the per-channel
element raw data $y$ as a parabolic Radon transform $G$ followed by a
convolution along the time axis with the pulse-echo impulse response $h$:

$$
y = \forw{x} = \left( h \ast_t G \right) \left( x \right).

$$

For each transmit event $k$, receive element $i$ and time sample
$t_n$, the sample is

$$
y_{k,i,n} = \left[h \ast_t G(x)\right]_{k,i,n}, \qquad
\left[G(x)\right]_{k,i,n} = \sum_{j} a_{k,i}(\mathbf{r}_j)\, K\!\big(f_s\,(t_n - \tau_{k,i}(\mathbf{r}_j))\big)\, x_j,

$$

where $\mathbf{r}_j = (x_j, z_j)$ is the position of pixel $j$,
$K$ is the linear interpolation kernel, $a_{k,i}$ is the product of
transmit and receive apodizations, and $\tau_{k,i}$ is the round-trip
time-of-flight

$$
\tau_{k,i}(x, z) = \frac{x \sin\theta_k + z \cos\theta_k}{c} + \frac{\|(x, z) - \mathbf{r}_i\|}{c}

$$

for the steering angle $\theta_k$.

The adjoint [`A_adjoint()`](#deepinv.physics.UltrasoundPlaneWave.A_adjoint) (also known as beamforming, or delay-and-sum in the special case of a Dirac pulse) follows the same formalism with a time-reversed pulse
$\tilde{h}(t) = h(-t)$ and the transpose quadratic Radon transform:

$$
\left[A^\top y\right]_j = \left[G^\top \! \left(\tilde{h} \ast_t y\right)\right]_j, \qquad
\left[G^\top y\right]_j = \sum_{k,i,n} a_{k,i}(\mathbf{r}_j)\, K\!\big(f_s\,(t_n - \tau_{k,i}(\mathbf{r}_j))\big)\, y_{k,i,n}.

$$

#### NOTE
We treat signals as real RF tensors: $x$ has shape `(B, 1, Z, X)` and $y$ has shape
`(B, 1, n_angles, n_elements, n_samples)`. If you would like to treat signals instead as complex IQ data, please open a feature request issue on GitHub.

#### NOTE
We interpolate time linearly. If you would like to interpolate with more advanced kernels, please open a feature request issue on GitHub.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – spatial image size `(Z, X)` in pixels.
  * **angles** (*Iterable* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – transmit steering angles in radians.
  * **element_positions** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – receive element positions in meters, shape `(n_elements, 2)` with columns `(x, z)`.
  * **n_samples** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of time samples recorded by each transducer element.
  * **sampling_frequency** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – sampling frequency in Hz.
  * **sound_speed** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – speed of sound $c$ in m/s. (default: `1540`)
  * **pixel_grid** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optional pixel positions in meters, shape
    `(Z, X, 2)` with columns `(x, z)`. If `None`, built from `pixel_size` and
    `pixel_origin`. Stored flattened, as the `(Z*X, 2)` buffer `pixel_grid`.
    (default: `None`)
  * **pixel_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – pixel spacing `(dz, dx)` in meters.
    (default: $c / (2 f_s)$ along both axes)
  * **pixel_origin** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – grid origin `(z0, x0)` in meters.
    (default: `(0, x_aperture_center)`)
  * **t0** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – acquisition-start offset $t_0$ in seconds,
    scalar or per-angle tensor of shape `(n_angles,)`. (default: `0`)
  * **f_number** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – receive f-number defining the aperture half-width
    $|x_i - x_j| \le z_j / f_\#$ at each pixel. `None` disables receive
    apodization. (default: `None`)
  * **receive_apod_window** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – receive apodization window inside the f-number
    aperture, one of `"rect"` or `"hann"`. Ignored if `f_number` is `None`.
    (default: `"rect"`)
  * **transmit_apod_window** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – transmit apodization window, one of `"rect"` or
    `"hann"`. `None` disables transmit apodization. (default: `None`)
  * **pulse** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optional real 1D pulse-echo impulse response $h$ (default: `None`)
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, [`A()`](#deepinv.physics.UltrasoundPlaneWave.A) and [`A_adjoint()`](#deepinv.physics.UltrasoundPlaneWave.A_adjoint) are divided by
    the operator’s spectral norm.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device for buffers. (default: `"cpu"`)

<hr />

* **Examples:**
  RF operator on a 32x32 image with 4 receive elements and 3 steering angles:
  ```pycon
  >>> import torch
  >>> from deepinv.physics import UltrasoundPlaneWave
  >>> ele_pos = torch.stack(
  ...     [torch.linspace(-1e-3, 1e-3, 4), torch.zeros(4)], dim=-1
  ... )
  >>> physics = UltrasoundPlaneWave(
  ...     img_size=(32, 32),
  ...     angles=torch.linspace(-0.28, 0.28, 3),
  ...     element_positions=ele_pos,
  ...     n_samples=256,
  ...     sampling_frequency=40e6,
  ...     sound_speed=1540.0,
  ...     t0=0.0,
  ...     normalize=False,
  ... )
  >>> x = torch.randn(1, 1, 32, 32)
  >>> physics(x).shape # 1, 1, n_angles, n_elements, n_samples)
  torch.Size([1, 1, 3, 4, 256])
  >>> physics.A_adjoint_A(x).shape # (1, 1, Z, X)
  torch.Size([1, 1, 32, 32])
  ```

  The transmit sequence can be changed in place with
  [`update_parameters()`](#deepinv.physics.UltrasoundPlaneWave.update_parameters):
  ```pycon
  >>> physics.update_parameters(angles=[0.0])  # keep a single transmit
  >>> physics(x).shape
  torch.Size([1, 1, 1, 4, 256])
  ```

#### A(x, \*\*kwargs)

Forward operator $y = \forw{x} = \left(h \ast_t G\right)(x)$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – image of shape `(B, 1, Z, X)`.
* **Returns:**
  RF per-channel raw data of shape `(B, 1, n_angles, n_elements, n_samples)`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, \*\*kwargs)

Adjoint (beamforming) operator $x = A^\top y = G^\top(\tilde{h} \ast_t y)$.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – raw RF data of shape `(B, 1, n_transmits, n_elements, n_samples)`.
* **Returns:**
  beamformed image of shape `(B, 1, Z, X)`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### update_parameters(angles=None, \*\*kwargs)

Update the transmit steering angles in place.

#### NOTE
Changing `angles` changes the number of transmits. If $t_0$ has all elements that are all identical, this is ok, otherwise you must rebuild operator instead.

* **Parameters:**
  **angles** (*Iterable* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – new transmit steering angles in radians.

<a id="sphx-glr-backref-deepinv-physics-ultrasoundplanewave"></a>

## Examples using `UltrasoundPlaneWave`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the plane-wave ultrafast ultrasound forward physics (deepinv.physics.UltrasoundPlaneWave) available in DeepInverse for pulse-echo imaging problems.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ultrasound_tour_thumb.png)

[Tour of ultrafast ultrasound in DeepInverse](https://deepinv.org/auto_examples/physics/demo_ultrasound_tour.html.md)

  <div class="sphx-glr-thumbnail-title">Tour of ultrafast ultrasound in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use deepinv.physics.UltrasoundPlaneWave to reconstructs an in-vivo carotid acquisition of the EPFL LTS5 ultrafast ultrasound dataset from raw RF ultrasound data.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_ultrasound_invivo_PnP_thumb.png)

[In-vivo ultrafast ultrasound reconstruction with Plug-and-Play](https://deepinv.org/auto_examples/plug-and-play/demo_ultrasound_invivo_PnP.html.md)

  <div class="sphx-glr-thumbnail-title">In-vivo ultrafast ultrasound reconstruction with Plug-and-Play</div>
</div>
<!-- thumbnail-parent-div-close --></div>
