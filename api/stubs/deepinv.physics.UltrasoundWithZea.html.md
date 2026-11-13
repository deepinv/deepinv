# UltrasoundWithZea

### *class* deepinv.physics.UltrasoundWithZea(pipeline, parameters, transmits_per_group=None, \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Ultrasound imaging operator with a [zea](https://zea.readthedocs.io/) beamforming pipeline.

General ultrasound physics where measurements `y` are IQ or RF raw channel data, and
images `x` are IQ or RF beamformed images.

Implemented with zea <sup>[1](#footcite-stevens2026zea)</sup>. The adjoint $A^{\top}$ is defined
by a closed-form differentiable beamforming pipeline (typically delay-and-sum (DAS)),
and the forward operator $A$ is computed with automatic differentiation.

Allows any type of ultrasound geometry and transmit scheme supported by [zea](https://zea.readthedocs.io/),
e.g. plane waves, diverging waves, focused lines, etc., as well as any linear
preprocessing that is part of the pipeline, such as demodulation and downsampling.

The acquisition (probe, transmits, pixel grid, …) is described by a
`zea.Parameters` object.

Measurements `y` have shape `(B, C, n_tx, n_ax, n_el)` where `C = n_ch`
(1 for RF, 2 for IQ data, 1 if `parameters` has no `n_ch`),
and the image `x` has shape `(B, C', ...)` e.g. `(B, 2, Z, X)` for an IQ
image.

#### TIP
Beamforming many transmits at once can use a lot of memory.
Use `transmits_per_group` to process transmits in groups sequentially.

#### NOTE
For a plane-wave operator with a closed-form forward operator, see
[`deepinv.physics.UltrasoundPlaneWave`](https://deepinv.org/api/stubs/deepinv.physics.UltrasoundPlaneWave.html.md#deepinv.physics.UltrasoundPlaneWave).

#### NOTE
This class requires `zea`. Install with
`pip install zea` and set `KERAS_BACKEND=torch` before importing zea.

#### NOTE
This physics does not yet support updating the zea pipeline, params or geometry
on the fly. If this feature is important for you, please raise a feature request
issue on GitHub.

* **Parameters:**
  * **pipeline** ([*zea.Pipeline*](https://zea.readthedocs.io/en/stable/_autosummary/zea.Pipeline.html#zea.Pipeline)) – zea linear beamforming (adjoint) pipeline that maps
    channel data to an image e.g. `zea.Pipeline([Demodulate(), Downsample(4), Beamform()])`.
  * **parameters** ([*zea.Parameters*](https://zea.readthedocs.io/en/stable/_autosummary/zea.Parameters.html#zea.Parameters)) – zea acquisition parameters. The transmits in
    `parameters.selected_transmits` correspond to the transmit axis of
    the measurements.
  * **transmits_per_group** ([*int*](https://docs.python.org/3.12/library/functions.html#int) *,* *None*) – number of transmits beamformed at once. If
    `None` (default), all transmits are beamformed at once.

<hr />

* **Examples:**
  With `params` a `zea.Parameters` and `y` RF channel data of shape
  `(B, 1, n_tx, n_ax, n_el)`:
  ```python
  from zea.ops import Beamform, Demodulate, Downsample

  pipeline = zea.Pipeline([Demodulate(), Downsample(factor=4), Beamform()])
  physics = dinv.physics.UltrasoundWithZea(pipeline, params)
  x_das = physics.A_adjoint(y)  # DAS image, (B, 2, Z, X)
  y_hat = physics.A(x_das)  # back to channel data, (B, 1, n_tx, n_ax, n_el)
  ```

<hr />

* **References:**

* <a id='footcite-stevens2026zea'>**[1]**</a> Tristan S. W. Stevens, Wessel L. van Nierop, Ben Luijten, Vincent van de Schaft, Oisín Nolan, Beatrice Federici, Louis D. van Harten, Simon W. Penninga, Noortje I. P. Schueler, and Ruud J. G. van Sloun. Zea: a toolbox for cognitive ultrasound imaging. *Journal of Open Source Software*, 11(121):9881, 2026.

#### A(x, \*\*kwargs)

Computes the channel data $Ax$, as the adjoint of the zea pipeline obtained
with automatic differentiation.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – image of shape `(B, C', ...)`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) channel data of shape `(B, C, n_tx, n_ax, n_el)`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint(y, \*\*kwargs)

Beamforms the channel data with the zea pipeline, i.e. computes $A^{\top}y$.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – channel data of shape `(B, C, n_tx, n_ax, n_el)`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) image of shape `(B, C', ...)`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-physics-ultrasoundwithzea"></a>

## Examples using `UltrasoundWithZea`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the zea ultrasound toolbox stevens2026zea with deepinv to form ultrasound images from raw channel data, by posing the image formation step, known as beamforming, as an inverse problem. We use the differentiable ultrasound processing pipeline of zea to compute the forward operator A of the linear model that relates the tissue reflectivity to the channel data, and solve the inverse problem with deepinv.optim.FISTA. First, a quick refresher on ultrasound image formation.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_zea_ultrasound_thumb.png)

[Ultrasound regularized beamforming with zea toolbox](https://deepinv.org/auto_examples/external-libraries/demo_zea_ultrasound.html.md)

  <div class="sphx-glr-thumbnail-title">Ultrasound regularized beamforming with zea toolbox</div>
</div>
<!-- thumbnail-parent-div-close --></div>
