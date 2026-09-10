# AnscombeDenoiser

### *class* deepinv.models.AnscombeDenoiser(denoiser, \*args, \*\*kwargs)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Wraps a Gaussian denoiser into a Poisson-Gaussian denoiser using the
[`Generalized Anscombe Transform (GAT)`](https://deepinv.org/api/stubs/deepinv.models.generalized_anscombe_transform.html.md#deepinv.models.generalized_anscombe_transform)
and its [`inverse`](https://deepinv.org/api/stubs/deepinv.models.inverse_generalized_anscombe_transform.html.md#deepinv.models.inverse_generalized_anscombe_transform).

Given a noisy observation $y$ corrupted by
[`Poisson-Gaussian noise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonGaussianNoise.html.md#deepinv.physics.PoissonGaussianNoise) with gain $\gamma$
and Gaussian standard deviation $\sigma$, the wrapper:

1. Applies the GAT to stabilize the variance to approximately $\gamma^2$:

$$
z = 2 \sqrt{\gamma y + \frac{3}{8}\gamma^2 + \sigma^2}
$$

1. Applies the wrapped Gaussian denoiser $\denoisername$ at noise level $\gamma$
   (which matches the standard deviation of the GAT output):

$$
\hat{z} = \denoisername(z,\; \sigma=\gamma)
$$

1. Applies the inverse GAT to return to the original domain.
   Setting $u = \hat{z}/\gamma$:

$$
\hat{y} = \gamma\left(\frac{1}{4}u^2 + \frac{1}{4}\sqrt{\frac{3}{2}}\,u^{-1} - \frac{11}{8}u^{-2} + \frac{5}{8}\sqrt{\frac{3}{2}}\,u^{-3} - \frac{1}{8} + \frac{\sigma^2}{\gamma^2}\right), \qquad u = \frac{\hat{z}}{\gamma}
$$

#### NOTE
When `gain = None` the noise is purely Gaussian and the GAT/IGAT are bypassed:
the wrapped denoiser is called directly as $\denoisername(y, \sigma)$.

* **Parameters:**
  **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – Gaussian denoiser $\denoisername$ to wrap.

<hr />

* **Examples:**
  ```pycon
  >>> import deepinv as dinv
  >>> import torch
  >>> from deepinv.models import AnscombeDenoiser, DRUNet
  >>> denoiser = DRUNet(pretrained="download")
  >>> anscombe_denoiser = AnscombeDenoiser(denoiser)
  >>> x = dinv.utils.load_example('butterfly.png')
  >>> y = dinv.physics.Denoising(dinv.physics.PoissonNoise(gain=.2))(x)
  >>> with torch.no_grad():
  ...     y_denoised = anscombe_denoiser(y, sigma=0., gain=0.2)
  >>> dinv.utils.plot([x, y, y_denoised], ["ground-truth", "noisy", "denoised"])
  ```

![image](../../_build/plot_directive/api/stubs/deepinv-models-AnscombeDenoiser-1.*)

#### forward(y, sigma, gain=None, \*args, \*\*kwargs)

Applies the Poisson-Gaussian denoiser via the Anscombe transform.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Noisy measurements corrupted by Poisson-Gaussian noise.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Standard deviation of the Gaussian noise $\sigma$.
  * **gain** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *|* *None*) – Gain of the Poisson distribution $\gamma$.
  * **args** – Additional positional arguments passed to the wrapped denoiser.
  * **kwargs** – Additional keyword arguments passed to the wrapped denoiser.
* **Return torch.Tensor:**
  Denoised measurements in the original domain.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-anscombedenoiser"></a>

## Examples using `AnscombeDenoiser`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise &lt;deepinv.physics.PoissonGaussianNoise&gt; using the Generalized Anscombe Transform (GAT) &lt;deepinv.models.AnscombeDenoiser&gt;, which converts any Gaussian denoiser into a Poisson-Gaussian denoiser :footcitemakitalo2012optimal.">  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div>
<!-- thumbnail-parent-div-close --></div>
