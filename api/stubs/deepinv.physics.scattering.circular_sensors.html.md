# circular_sensors

### deepinv.physics.scattering.circular_sensors(number, radius, max_angle=360, offset_angle=0, device='cpu')

Generate equispaced sensors on a circle.

* **Parameters:**
  * **number** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of sensors.
  * **radius** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Radius of the circle.
  * **max_angle** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Maximum angle in degrees covered by sensors.
  * **offset_angle** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Offset angle in degrees.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Torch device for tensors.
* **Returns:**
  Tuple of tensors:
  - `transmitters`: Tensor of shape `(2, number)` with (x,y) positions.
  - `receivers`: Tensor of shape `(2, number, number-1)` with (x,y) positions.

## Examples using `circular_sensors`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we show how to use the deepinv.physics.Scattering forward model.">  <div class="sphx-glr-thumbnail-title">Inverse scattering problem</div>
</div>
<!-- thumbnail-parent-div-close --></div>
