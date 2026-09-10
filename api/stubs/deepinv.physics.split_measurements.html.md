# split_measurements

### deepinv.physics.split_measurements(y, physics, num_subsets)

Splits tomography measurements into angular subsets.

#### NOTE
The expected measurement layout depends on the tomography physics used:

* [`deepinv.physics.Tomography`](https://deepinv.org/api/stubs/deepinv.physics.Tomography.html.md#deepinv.physics.Tomography): `[B, C, N, A]`, where `A`
  is the angle axis and `N` is the detector axis.
* [`deepinv.physics.TomographyWithAstra`](https://deepinv.org/api/stubs/deepinv.physics.TomographyWithAstra.html.md#deepinv.physics.TomographyWithAstra): `[B, C, A, N]` in
  2D and `[B, C, V, A, N]` in 3D, where `V` and `N` are the
  detector axes.
* [`deepinv.physics.PET`](https://deepinv.org/api/stubs/deepinv.physics.PET.html.md#deepinv.physics.PET): `[B, C, N, A]` in 2D and
  `[B, C, N, A, P]` in 3D for the default RVP sinogram order, where
  `C = 1`, `N` is the radial detector axis, `A` is the view
  axis, and `P` is the plane axis.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – full measurement tensor.
  * **physics** ([*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – tomography physics.
  * **num_subsets** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of subsets.
* **Returns:**
  measurements as a [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList).
* **Return type:**
  [*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)

## Examples using `split_measurements`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div>
<!-- thumbnail-parent-div-close --></div>
