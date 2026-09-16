# RicianNoise

### *class* deepinv.physics.RicianNoise(sigma=0.1, rng=None)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Rician noise: $y = \sqrt{(x + \sigma \epsilon_1)^2 + (\sigma \epsilon_2)^2}$

where $\epsilon_1\sim\mathcal{N}(0,I)$ and $\epsilon_2\sim\mathcal{N}(0,I)$

This noise model is often used in MRI imaging and has the property of keeping pixel intensities $\geq 0$

#### WARNING
All pixel intensities will become positive: this noise model may not be suited for data with negative intensities.

* **Parameters:**
  * **sigma** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – Standard deviation used.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.

#### forward(x, sigma=None, seed=None)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – standard deviation to be used.
    If not `None`, it will overwrite the current noise level.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – the seed for the random number generator.
* **Returns:**
  noisy measurements
