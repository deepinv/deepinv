# UniformNoise

### *class* deepinv.physics.UniformNoise(a=0.1, rng=None)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Uniform noise $y = x + \epsilon$ where $\epsilon\sim\mathcal{U}(-a,a)$.

* **Parameters:**
  * **a** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *]*) – amplitude of the noise.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.

<hr />

* **Examples:**
  Adding uniform noise to a physics operator by setting the `noise_model`
  attribute of the physics operator:
  ```pycon
  >>> from deepinv.physics import Denoising, UniformNoise
  >>> import torch
  >>> physics = Denoising()
  >>> physics.noise_model = UniformNoise()
  >>> x = torch.rand(1, 1, 2, 2)
  >>> y = physics(x)
  ```

#### forward(x, a=None, seed=None, \*\*kwargs)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **a** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – amplitude of the noise. If not None, it will overwrite the current noise level.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.
* **Returns:**
  noisy measurements

<a id="sphx-glr-backref-deepinv-physics-uniformnoise"></a>

## Examples using `UniformNoise`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div>
<!-- thumbnail-parent-div-close --></div>
