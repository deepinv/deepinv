# VariancePreservingDiffusion

### *class* deepinv.sampling.VariancePreservingDiffusion(denoiser=None, beta_min=0.1, beta_max=20.0, alpha=0.0, T=1.0, scaled_linear=False, solver=None, sigma_t=None, sigma_prime_t=None, dtype=torch.float64, device=torch.device('cpu'), \*args, \*\*kwargs)

Bases: [`SongDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.SongDiffusionSDE.html.md#deepinv.sampling.SongDiffusionSDE)

Variance-Preserving Stochastic Differential Equation (VP-SDE).

This class implements the reverse-time SDE of a Variance-Preserving forward-time noising process Song *et al.*<sup>[1](#footcite-song2020score)</sup>.

The generative process (reverse-time SDE) can be defined equivalently in two ways:

- as a special case of [`deepinv.sampling.SongDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.SongDiffusionSDE.html.md#deepinv.sampling.SongDiffusionSDE) with $\beta(t) = \xi(t)$:
  $$
  d x_t = -\frac{1}{2} \beta(t) \left( x_t + \left(1 + \alpha(t) \right) \nabla \log p_t(x_t) \right) dt + \sqrt{\alpha(t) \beta(t)} d w_t

  $$
- or as a special case of [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE) with the rescaling schedule defined with respect to the noise schedule by
  $s(t) = \left(1 + \sigma(t)^2\right)^{-1/2}$, giving:
  $$
  d x_t = -\frac{\sigma(t) \sigma'(t)}{1 + \sigma(t)^2} \left( x_t + \left(1 + \alpha(t) \right) \nabla \log p_t(x_t) \right) dt + \sqrt{\frac{2 \alpha(t) \sigma(t) \sigma'(t)}{1 + \sigma(t)^2}} d w_t

  $$

Both formulations are equivalent, and are linked via the relation

$$
\beta(t) = \frac{2 \sigma(t) \sigma'(t)}{1 + \sigma(t)^2}.

$$

By default, we use the first formulation with the linear schedule for $\beta(t)$ orginally proposed in Song *et al.*<sup>[1](#footcite-song2020score)</sup>:

$$
\beta(t) = \beta_{\mathrm{min}} + t \left( \beta_{\mathrm{max}} - \beta_{\mathrm{min}} \right),
$$

where, by default, $\beta_{\mathrm{min}} = 0.1$ and $\beta_{\mathrm{max}} = 20$.

We also provide the option to define a custom noise schedule $\sigma(t)$ by providing the argument `sigma_t`.
In this case, the $\beta(t)$ schedule is ignored and the rescaling schedule is automatically set to $s(t) = \left(1 + \sigma(t)^2\right)^{-1/2}$.

#### NOTE
This SDE must be solved going reverse in time i.e. from $t=T$ to $t=0$.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser used to provide an approximation of the score at time $t$: $\nabla \log p_t$.
  * **beta_min** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the value $\beta_{\mathrm{min}} = \beta(0)$ of the linear schedule. Default to `0.1`.
  * **beta_max** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the value $\beta_{\mathrm{max}} = \beta(1)$ of the linear schedule. Default to `20.0`.
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.12/library/functions.html#float)) – a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function $\alpha(t) = 0$ corresponds to ODE sampling and $\alpha(t) > 0$ corresponds to SDE sampling.
  * **T** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the end time of the forward SDE. Default to `1.0`.
  * **scaled_linear** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – whether to use the scaled linear schedule, which is linear in $\sqrt{\beta(t)}$ instead of $\beta(t)$. Default to `False`.
  * **solver** ([*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver for solving the SDE.
  * **sigma_t** (*Callable*) – a custom noise level $\sigma(t)$, replacing `beta_min`, `beta_max` and `scaled_linear`, e.g. `lambda t: 80 * t**2`.
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

<a id="sphx-glr-backref-deepinv-sampling-variancepreservingdiffusion"></a>

## Examples using `VariancePreservingDiffusion`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares three diffusion posterior samplers on a noisy inpainting problem:">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_samplers_thumb.png)

[Posterior sampling with DPS, DiffPIR and DDRM](https://deepinv.org/auto_examples/sampling/demo_diffusion_samplers.html.md)

  <div class="sphx-glr-thumbnail-title">Posterior sampling with DPS, DiffPIR and DDRM</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_sde_thumb.png)

[Building your diffusion posterior sampling method using SDEs](https://deepinv.org/auto_examples/sampling/demo_diffusion_sde.html.md)

  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div>
<!-- thumbnail-parent-div-close --></div>
