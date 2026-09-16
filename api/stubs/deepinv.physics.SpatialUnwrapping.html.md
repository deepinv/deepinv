# SpatialUnwrapping

### *class* deepinv.physics.SpatialUnwrapping(threshold=1.0, mode='round', \*\*kwargs)

Bases: [`Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

Spatial unwrapping forward operator.

This class implements a forward operator for spatial unwrapping, where the input is wrapped modulo a threshold value.
The operator can use either floor or round mode for the wrapping operation. It is useful for problems where the observed data is wrapped,
such as in phase imaging, modulo imaging, or interferometry.

The forward operator is defined as:

$$
y = w_t(x) = x - t \cdot \mathrm{q}(x / t)
$$

where $w_t$ is the modulo operator, $t$ is the threshold, and $\mathrm{q}$ is either the rounding or flooring function depending on the mode.

* **Parameters:**
  * **threshold** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – The threshold $t$ value for the modulo operation (default: 1.0).
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – modulo function $q(\cdot)$, either ‘round’ or ‘floor’ (default: ‘round’).
  * **kwargs** – Additional arguments passed to the base Physics class.

<hr />

* **Example:**
  ```pycon
  >>> import torch
  >>> from deepinv.physics.spatial_unwrapping import SpatialUnwrapping
  >>> x = torch.tensor([[0.5, 1.2, 2.7]])
  >>> physics = SpatialUnwrapping(threshold=1.0, mode="round")
  >>> y = physics(x)
  >>> print(torch.round(y, decimals=1))
  tensor([[ 0.5000,  0.2000, -0.3000]])
  ```

#### A(x, \*\*kwargs)

Applies the modulo operator to the input tensor.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) Modulo tensor.

#### A_adjoint(y, \*\*kwargs)

Adjoint operator for the modulo operator. For the modulo operator, the adjoint is the identity.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) Output tensor (identity).

#### forward(x, \*\*kwargs)

Applies the forward model for spatial unwrapping.

In spatial unwrapping, the noise model is first applied to the input, followed by the modulo operator.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) The result after applying noise, modulo operator, and sensor.

<a id="sphx-glr-backref-deepinv-physics-spatialunwrapping"></a>

## Examples using `SpatialUnwrapping`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div>
<!-- thumbnail-parent-div-close --></div>
