# LaplaceNoise

### *class* deepinv.physics.LaplaceNoise(b=0.1, rng=None)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Laplace noise $y = z + \epsilon$ where $\epsilon\sim\text{Laplace}(0,b)$.
In the laplace distribution, b is the scale parameter and the variance is given by $\sigma^2=2b^2$.

* **Parameters:**
  * **b** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *]*) – scale of the noise.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.

<hr />
:Examples:

> Adding Laplace noise to a physics operator by setting the `noise_model`
> attribute of the physics operator:

> ```pycon
> >>> from deepinv.physics import Denoising, LaplaceNoise
> >>> import torch
> >>> physics = Denoising()
> >>> physics.noise_model = LaplaceNoise()
> >>> x = torch.rand(1, 1, 2, 2)
> >>> y = physics(x)
> ```

#### forward(x, b=None, seed=None, \*\*kwargs)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **b** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – scale of the noise. If not None, it will overwrite the current noise level.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.
* **Returns:**
  noisy measurements
