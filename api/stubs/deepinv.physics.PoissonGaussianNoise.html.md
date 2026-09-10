# PoissonGaussianNoise

### *class* deepinv.physics.PoissonGaussianNoise(gain=1.0, sigma=0.1, clip_positive=False, min_gain=1e-12, rng=None)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Poisson-Gaussian noise $y = \gamma z + \epsilon$ where $z\sim\mathcal{P}(\frac{x}{\gamma})$
and $\epsilon\sim\mathcal{N}(0, I \sigma^2)$.

This noise model allows to recover the Poisson noise model by setting the standard deviation to zero,
i.e., $\sigma=0$, and the Gaussian noise model by setting the gain to zero, i.e., $\gamma\to0$.

#### NOTE
If $\gamma=0$, the model will clamp the input to a small value
to avoid division by zero, i.e., $\gamma=\max(\gamma, \text{min\_gain})$.

#### TIP
All [pretrained denoisers](https://deepinv.org/user_guide/reconstruction/denoisers.html.md#denoisers) in the library can be re-used for Poisson-Gaussian denoising
using the [`Anscombe transform`](https://deepinv.org/api/stubs/deepinv.models.AnscombeDenoiser.html.md#deepinv.models.AnscombeDenoiser).

* **Parameters:**
  * **gain** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – gain of the noise.
  * **sigma** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – Standard deviation of the noise.
  * **clip_positive** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – (optional) if True, the input is clipped to be positive before adding noise.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.

<hr />

* **Examples:**
  Adding Poisson gaussian noise to a physics operator by setting the `noise_model`
  attribute of the physics operator:
  ```pycon
  >>> from deepinv.physics import Denoising, PoissonGaussianNoise
  >>> import torch
  >>> physics = Denoising()
  >>> physics.noise_model = PoissonGaussianNoise()
  >>> x = torch.rand(1, 1, 2, 2)
  >>> y = physics(x)
  ```

#### forward(x, gain=None, sigma=None, seed=None, \*\*kwargs)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **gain** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – gain of the noise. If not None, it will overwrite the current gain.
  * **sigma** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor containing gain and standard deviation.
    If not None, it will overwrite the current gain and standard deviation.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.
* **Returns:**
  noisy measurements

<a id="sphx-glr-backref-deepinv-physics-poissongaussiannoise"></a>

## Examples using `PoissonGaussianNoise`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise &lt;deepinv.physics.PoissonGaussianNoise&gt; using the Generalized Anscombe Transform (GAT) &lt;deepinv.models.AnscombeDenoiser&gt;, which converts any Gaussian denoiser into a Poisson-Gaussian denoiser :footcitemakitalo2012optimal.">  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
