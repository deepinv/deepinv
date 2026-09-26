# generalized_anscombe_transform

### deepinv.models.generalized_anscombe_transform(x, gain, sigma)

Generalized Anscombe Transform (GAT)

The transform converts a noisy observation $y$ from a [`Poisson-Gaussian distribution`](https://deepinv.org/api/stubs/deepinv.physics.PoissonGaussianNoise.md#deepinv.physics.PoissonGaussianNoise) with
gain $\gamma$ and [`Gaussian noise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.md#deepinv.physics.GaussianNoise) standard deviation $\sigma$ to an approximately Gaussian
distribution with variance $\gamma$, see Makitalo and Foi<sup>[1](#footcite-makitalo2012optimal)</sup>.

The transform is defined as:

$$
h(y) = 2 \sqrt{\gamma x + \frac{3}{8}\gamma^2 + \sigma^2}
$$

#### NOTE
The formula varies slightly from the one proposed by Makitalo and Foi<sup>[1](#footcite-makitalo2012optimal)</sup>,
as the library considers a normalized Poisson-Gaussian noise model, $y = \gamma \mathcal{P}(x/\gamma) + \epsilon$,
whereas the authors consider $y = \gamma \mathcal{P}(x) + \epsilon$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – tensor corrupted with Poisson-Gaussian noise
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Gain of the Poisson distribution $\gamma$
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Standard deviation of the Gaussian noise $\sigma$
* **Return torch.Tensor:**
  Transformed measurements

<hr />

* **References:**

* <a id='footcite-makitalo2012optimal'>**[1]**</a> Markku Makitalo and Alessandro Foi. Optimal inversion of the generalized anscombe transformation for poisson-gaussian noise. *IEEE transactions on image processing*, 22(1):91–103, 2012.

## Examples using `generalized_anscombe_transform`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise using the Generalized Anscombe Transform (GAT), which converts any Gaussian denoiser into a Poisson-Gaussian denoiser makitalo2012optimal.">![](auto_examples/physics/images/thumb/sphx_glr_demo_anscombe_thumb.png)

[Poisson-Gaussian Denoising with the Generalized Anscombe Transform](https://deepinv.org/auto_examples/physics/demo_anscombe.md)

  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div>
<!-- thumbnail-parent-div-close --></div>
