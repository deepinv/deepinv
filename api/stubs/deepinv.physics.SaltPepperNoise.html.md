# SaltPepperNoise

### *class* deepinv.physics.SaltPepperNoise(p=0.025, s=0.025, rng=None)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

SaltPepper noise $y = \begin{cases} 0 & \text{if } z < p\\ z & \text{if } z \in [p, 1-s]\\ 1 & \text{if } z > 1 - s\end{cases}$ with $z\sim\mathcal{U}(0,1)$

This noise model is also known as impulse noise, is a form of noise sometimes seen on digital images.
For black-and-white or grayscale images, it presents as sparsely occurring white and black pixels,
giving the appearance of an image sprinkled with salt and pepper.

The parameters s and p control the amount of salt (pixel to 1) and pepper (pixel to 0) noise.

* **Parameters:**
  * **s** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – amount of salt noise.
  * **p** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – amount of pepper noise.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.

<hr />

* **Examples:**
  Adding LogPoisson noise to a physics operator by setting the `noise_model`
  attribute of the physics operator:
  ```pycon
  >>> from deepinv.physics import Denoising, SaltPepperNoise
  >>> import torch
  >>> physics = Denoising()
  >>> physics.noise_model = SaltPepperNoise()
  >>> x = torch.rand(1, 1, 2, 2)
  >>> y = physics(x)
  ```

#### forward(x, p=None, s=None, seed=None, \*\*kwargs)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **s** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – amount of salt noise.
    If not None, it will overwrite the current salt noise.
  * **p** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – amount of pepper noise.
    If not None, it will overwrite the current pepper noise.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.
* **Returns:**
  noisy measurements
