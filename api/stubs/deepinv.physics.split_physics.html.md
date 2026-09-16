# split_physics

### deepinv.physics.split_physics(physics, num_subsets, device)

Builds a stacked tomography physics with one operator per angular subset.

#### WARNING
If `physics` is normalized, each subset reuses the operator norm of the
complete physics instead of computing its own.
Computing the real subset physics operator norm would result in a mismatch
between the projections of the full physics and the subset physics.

* **Parameters:**
  * **physics** ([*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – tomography physics.
  * **num_subsets** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of subsets.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device on which to create the subset physics.
* **Returns:**
  [`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics) over angular subsets.
* **Return type:**
  [*StackedLinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics)

## Examples using `split_physics`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div>
<!-- thumbnail-parent-div-close --></div>
