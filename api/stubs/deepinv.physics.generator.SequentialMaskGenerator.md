# SequentialMaskGenerator

### *class* deepinv.physics.generator.SequentialMaskGenerator(spatial_generator, T, \*\*kwargs)

Bases: [`PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.md#deepinv.physics.generator.PhysicsGenerator)

Let a mask generator generate non-overlapping sequential sampling patterns.

Given a generator that generates static masks (i.e. without a time dimension), split the samples into
`T` time-steps such that the original mask is the temporal union of all resulting time-steps.
The resulting mask is “time-resolved” and simulates shot-based acquisition sequences.

* **Parameters:**
  * **spatial_generator** ([*BaseMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.md#deepinv.physics.generator.BaseMaskGenerator)) – static Cartesian mask generator.
  * **T** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of time-steps to split the mask into.

#### step(batch_size=1, seed=None, \*\*kwargs)

Generate time-resolved sequential mask from the sampling mask.

The $n$ sampled lines are split in sequential order into chunks of $\lceil n/T \rceil$ lines
i.e. the $k$-th sampled line is assigned to time-step $\lfloor k / \lceil n/T \rceil \rfloor$.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – batch size.
  * **seed** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – optional seed for the random number generator.
* **Returns:**
  dictionary with key **‘mask’** of shape `(batch_size, C, T, H, W)`.
* **Return type:**
  [dict](https://docs.python.org/3.12/library/stdtypes.html#dict)

<a id="sphx-glr-backref-deepinv-physics-generator-sequentialmaskgenerator"></a>

## Examples using `SequentialMaskGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many imaging modalities are not acquired all at once but sequentially in &quot;shots&quot;, e.g. kspace lines in MRI or projections in CT. Since the time between shots is non-negligible, the subject may move during scanning, and reconstructions that assume a static subject will be motion-corrupted. Modelling a transform per shot allows us to simulate this motion corruption and, given the motion parameters, to compensate for it during reconstruction.">![](auto_examples/physics/images/thumb/sphx_glr_demo_motion_mri_thumb.png)

[Motion simulation and motion-compensated reconstruction](https://deepinv.org/auto_examples/physics/demo_motion_mri.md)

  <div class="sphx-glr-thumbnail-title">Motion simulation and motion-compensated reconstruction</div>
</div>
<!-- thumbnail-parent-div-close --></div>
