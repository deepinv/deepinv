# BrownianGenerator

### *class* deepinv.physics.generator.BrownianGenerator(param_key, T, dt=0.04, sigma=0.75, bound=3.0, \*\*kwargs)

Bases: [`TrajectoryGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.TrajectoryGenerator.html.md#deepinv.physics.generator.TrajectoryGenerator)

Generates Brownian motion trajectories.

The trajectory $p_t$ starts at 0. It is generated as

$$
p_{t+1} = p_{t} + \sigma \sqrt{dt} \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 1),

$$

and is reflected at $\pm$ `bound`.

* **Parameters:**
  * **param_key** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – name of the generated parameter.
  * **T** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of time steps.
  * **dt** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – time between steps in seconds.
  * **sigma** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – diffusion per square-root second.
  * **bound** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – maximum absolute value of the trajectory.

<hr />

* **Example:**

```pycon
>>> from deepinv.physics.generator import BrownianGenerator
>>> generator = BrownianGenerator("theta", T=10) + BrownianGenerator("shift", T=10)
>>> generator.step(batch_size=2)["theta"].shape
torch.Size([2, 10])
```

<a id="sphx-glr-backref-deepinv-physics-generator-browniangenerator"></a>

## Examples using `BrownianGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many imaging modalities are not acquired all at once but sequentially in &quot;shots&quot;, e.g. kspace lines in MRI or projections in CT. Since the time between shots is non-negligible, the subject may move during scanning, and reconstructions that assume a static subject will be motion-corrupted. Modelling a transform per shot allows us to simulate this motion corruption and, given the motion parameters, to compensate for it during reconstruction.">![](auto_examples/physics/images/thumb/sphx_glr_demo_motion_mri_thumb.png)

[Motion simulation and motion-compensated reconstruction](https://deepinv.org/auto_examples/physics/demo_motion_mri.html.md)

  <div class="sphx-glr-thumbnail-title">Motion simulation and motion-compensated reconstruction</div>
</div>
<!-- thumbnail-parent-div-close --></div>
