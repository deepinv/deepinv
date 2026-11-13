# TrajectoryGenerator

### *class* deepinv.physics.generator.TrajectoryGenerator(param_key, T, rng=None, device='cpu', dtype=torch.float32)

Bases: [`PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.md#deepinv.physics.generator.PhysicsGenerator)

Base generator for parameter trajectories over time.

Generates trajectories of shape `(batch_size, T, ...)` of parameters, to be used for
specifying transforms varying across time.

#### TIP
See [Dynamic & time-resolved imaging](https://deepinv.org/user_guide/physics/physics.md#dynamic-imaging) for example usage of trajectory generators with a `transform` for modelling transform trajectories.

* **Parameters:**
  * **param_key** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – name of the generated parameter.
  * **T** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of time steps.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – torch random generator.
  * **device** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device of the generated trajectories.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the generated trajectories.

#### sample_trajectory(batch_size)

Sample trajectories. This must be implemented in child classes.

* **Parameters:**
  **batch_size** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of trajectories.
* **Return torch.Tensor:**
  trajectories of shape `(batch_size, T, ...)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### step(batch_size=1, seed=None, \*\*kwargs)

Generate a batch of trajectories.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of trajectories.
  * **seed** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – optional random seed.
* **Returns:**
  dictionary with key `param_key` of shape `(batch_size, T, ...)`.
* **Return type:**
  [dict](https://docs.python.org/3.12/library/stdtypes.html#dict)

<a id="sphx-glr-backref-deepinv-physics-generator-trajectorygenerator"></a>

## Examples using `TrajectoryGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many imaging modalities are not acquired all at once but sequentially in &quot;shots&quot;, e.g. kspace lines in MRI or projections in CT. Since the time between shots is non-negligible, the subject may move during scanning, and reconstructions that assume a static subject will be motion-corrupted. Modelling a transform per shot allows us to simulate this motion corruption and, given the motion parameters, to compensate for it during reconstruction.">![](auto_examples/physics/images/thumb/sphx_glr_demo_motion_mri_thumb.png)

[Motion simulation and motion-compensated reconstruction](https://deepinv.org/auto_examples/physics/demo_motion_mri.md)

  <div class="sphx-glr-thumbnail-title">Motion simulation and motion-compensated reconstruction</div>
</div>
<!-- thumbnail-parent-div-close --></div>
