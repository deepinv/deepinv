# VarianceExplodingDiffusion

### *class* deepinv.sampling.VarianceExplodingDiffusion(denoiser=None, sigma_min=0.001, sigma_max=80, alpha=0.25, T=1.0, solver=None, sigma_t=None, sigma_prime_t=None, dtype=torch.float64, device=torch.device('cpu'), \*args, \*\*kwargs)

Bases: [`EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE)

Variance-Exploding Stochastic Differential Equation (VE-SDE).

This class implements the reverse-time SDE of a Variance Exploding forward-time noising process Song *et al.*<sup>[1](#footcite-song2020score)</sup>.

The generative process (reverse-time SDE) is defined as follows:

$$
d x_t = -\left(1 + \alpha(t) \right) \sigma(t) \sigma'(t) \nabla \log p_t(x_t) dt + \sqrt{2 \alpha(t) \sigma(t) \sigma'(t)} d w_t

$$

where the time-dependent noise level $\sigma(t)$ is defined by the argument `sigma_t`.
It corresponds to [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE) with a constant scaling $s(t) = 1$.

By default, we take the geometric noise schedule,  as proposed originally in Song *et al.*<sup>[1](#footcite-song2020score)</sup>.

$$
\sigma(t) = \sigma_{\mathrm{min}} \left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}} \right)^t,
$$

#### NOTE
This SDE must be solved going reverse in time i.e. from $t=T$ to $t=0$.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser used to provide an approximation of the score at time $t$: $\nabla \log p_t$.
  * **sigma_min** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the minimum noise level.
  * **sigma_max** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the maximum noise level.
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.12/library/functions.html#float)) – a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function $\alpha(t) = 0$ corresponds to ODE sampling and $\alpha(t) > 0$ corresponds to SDE sampling.
  * **T** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the end time of the forward SDE. Default to `1.0`.
  * **solver** ([*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver for solving the SDE.
  * **sigma_t** (*Callable*) – a custom noise level $\sigma(t)$, replacing `sigma_min` and `sigma_max`, e.g. `lambda t: 80 * t**2`.
  * **sigma_prime_t** (*Callable*) – the derivative $\sigma'(t)$ of `sigma_t`. If `None` (default), it is computed by autograd and it is ignored if `sigma_t` is not provided.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the computation, except for the `denoiser` which will use `torch.float32`.
    We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
    most computation cost is from evaluating the `denoiser`, which will be always computed in `torch.float32`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device on which the computation is performed.
  * **\*args** – additional arguments for the [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).
  * **\*\*kwargs** – additional keyword arguments for the [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).

<hr />

* **References:**

* <a id='footcite-song2020score'>**[1]**</a> Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*. 2020.

<a id="sphx-glr-backref-deepinv-sampling-varianceexplodingdiffusion"></a>

## Examples using `VarianceExplodingDiffusion`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusers_thumb.png)

[Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse](https://deepinv.org/auto_examples/sampling/demo_diffusers.html.md)

  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares three diffusion posterior samplers on a noisy inpainting problem:">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_samplers_thumb.png)

[Posterior sampling with DPS, DiffPIR and DDRM](https://deepinv.org/auto_examples/sampling/demo_diffusion_samplers.html.md)

  <div class="sphx-glr-thumbnail-title">Posterior sampling with DPS, DiffPIR and DDRM</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_sde_thumb.png)

[Building your diffusion posterior sampling method using SDEs](https://deepinv.org/auto_examples/sampling/demo_diffusion_sde.html.md)

  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares eight approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.html.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
