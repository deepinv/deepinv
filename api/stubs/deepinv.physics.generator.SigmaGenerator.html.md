# SigmaGenerator

### *class* deepinv.physics.generator.SigmaGenerator(sigma_min=0.01, sigma_max=0.5, rng=None, device='cpu', dtype=torch.float32)

Bases: [`PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)

Generator for the noise level $\sigma$ in the [`Gaussian noise model`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise).

The noise level is sampled uniformly from the interval $[\sigma_{\text{min}}, \sigma_{\text{max}}]$.

* **Parameters:**
  * **sigma_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – minimum noise level
  * **sigma_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum noise level
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device where the tensor is stored
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the generated tensor.

<hr />

* **Examples:**

```pycon
>>> from deepinv.physics.generator import SigmaGenerator
>>> generator = SigmaGenerator()
>>> sigma_dict = generator.step(seed=0) # dict_keys(['sigma'])
>>> print(sigma_dict['sigma'])
tensor([0.2532])
```

#### step(batch_size=1, seed=None, \*\*kwargs)

Generates a batch of noise levels.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batch size
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.
* **Returns:**
  dictionary with key **‘sigma’**: tensor of size (batch_size,).
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

<a id="sphx-glr-backref-deepinv-physics-generator-sigmagenerator"></a>

## Examples using `SigmaGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div>
<!-- thumbnail-parent-div-close --></div>
