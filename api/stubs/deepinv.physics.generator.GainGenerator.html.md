# GainGenerator

### *class* deepinv.physics.generator.GainGenerator(gain_min=0.1, gain_max=0.4, rng=None, device='cpu', dtype=torch.float32)

Bases: [`PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)

Generator for the noise level $\gamma$ in the [`Poisson noise model`](https://deepinv.org/api/stubs/deepinv.physics.PoissonNoise.html.md#deepinv.physics.PoissonNoise).

The gain is sampled uniformly from the interval $[\gamma_\text{min}, \gamma_\text{max}]$.

* **Parameters:**
  * **gain_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – minimum noise level
  * **gain_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum noise level
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device where the tensor is stored
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the generated tensor.

<hr />

* **Examples:**

```pycon
>>> from deepinv.physics.generator import GainGenerator
>>> generator = GainGenerator()
>>> params = generator.step(seed=0) # params(['gain'])
>>> print(params['gain'])
tensor([0.2489])
```

#### step(batch_size=1, seed=None, \*\*kwargs)

Generates a batch of noise levels.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batch size
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.
* **Returns:**
  dictionary with key **‘gain’**: tensor of size (batch_size,).
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)
