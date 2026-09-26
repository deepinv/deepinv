# UniformGaussianNoise

### *class* deepinv.physics.UniformGaussianNoise(sigma_min=0.0, sigma_max=0.5, rng=None)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Gaussian noise $y=z+\epsilon$ where
$\epsilon\sim \mathcal{N}(0,I\sigma^2)$ and
$\sigma \sim\mathcal{U}(\sigma_{\text{min}}, \sigma_{\text{max}})$

* **Parameters:**
  * **sigma_min** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – minimum standard deviation of the noise.
  * **sigma_max** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – maximum standard deviation of the noise.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.

<hr />

* **Examples:**
  Adding uniform gaussian noise to a physics operator by setting the `noise_model`
  attribute of the physics operator:
  ```pycon
  >>> from deepinv.physics import Denoising, UniformGaussianNoise
  >>> import torch
  >>> physics = Denoising()
  >>> physics.noise_model = UniformGaussianNoise()
  >>> x = torch.rand(1, 1, 2, 2)
  >>> y = physics(x)
  ```

#### forward(x, seed=None, \*\*kwargs)

Adds the noise to measurements x.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.
* **Returns:**
  noisy measurements.
