# PoissonNoise

### *class* deepinv.physics.PoissonNoise(gain=1.0, normalize=True, clip_positive=False, rng=None)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Poisson noise $y = \mathcal{P}(\frac{x}{\gamma})$
with gain $\gamma>0$.

If `normalize=True`, the output is multiplied by the gain, i.e., $\tilde{y} = \gamma y$.

* **Parameters:**
  * **gain** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – gain of the noise.
  * **normalize** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – normalize the output.
  * **clip_positive** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – clip the input to be positive before adding noise.
    This may be needed when a NN outputs negative values e.g. when using leaky ReLU.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.

#### NOTE
Poisson noise is only defined for non-negative inputs.
When used in combination with physics operators that can produce negative outputs (such as [`deepinv.physics.BlurFFT`](https://deepinv.org/api/stubs/deepinv.physics.BlurFFT.html.md#deepinv.physics.BlurFFT)), it is recommended to set `clip_positive=True` to avoid runtime errors.

#### TIP
All [pretrained denoisers](https://deepinv.org/user_guide/reconstruction/denoisers.html.md#denoisers) in the library can be re-used for Poisson denoising
using the [`Anscombe transform`](https://deepinv.org/api/stubs/deepinv.models.AnscombeDenoiser.html.md#deepinv.models.AnscombeDenoiser).

<hr />

* **Examples:**
  Adding Poisson noise to a physics operator by setting the `noise_model`
  attribute of the physics operator:
  ```pycon
  >>> from deepinv.physics import Denoising, PoissonNoise
  >>> import torch
  >>> physics = Denoising()
  >>> physics.noise_model = PoissonNoise()
  >>> x = torch.rand(1, 1, 2, 2)
  >>> y = physics(x)
  ```

#### forward(x, gain=None, seed=None, \*\*kwargs)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **gain** (*None* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – gain of the noise. If not None, it will overwrite the current noise level.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.
* **Returns:**
  noisy measurements

<a id="sphx-glr-backref-deepinv-physics-poissonnoise"></a>

## Examples using `PoissonNoise`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise &lt;deepinv.physics.PoissonGaussianNoise&gt; using the Generalized Anscombe Transform (GAT) &lt;deepinv.models.AnscombeDenoiser&gt;, which converts any Gaussian denoiser into a Poisson-Gaussian denoiser :footcitemakitalo2012optimal.">  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
