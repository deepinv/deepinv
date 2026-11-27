# AncestralSolver

### *class* deepinv.sampling.AncestralSolver(timesteps=None, t_start=None, t_end=None, num_steps=None, eta=None, variance='small', rng=None)

Bases: [`BaseSDESolver`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)

Ancestral solver for reverse-time diffusion SDEs, generalizing the DDPM and DDIM samplers.

It solves the reverse-time SDE (see [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE)), from $t = T$ to $t = 0$:

$$
d x_t = \left(\frac{s'(t)}{s(t)} x_t - (1 + \alpha(t)) s(t)^2 \sigma(t) \sigma'(t) \nabla \log p_t(x_t) \right) dt + s(t) \sqrt{2 \alpha(t) \sigma(t) \sigma'(t)} d w_t.

$$

On a step from $t$ to $t + dt$ (with $dt < 0$ for reverse-time sampling), the solver computes the next state $x_{t+dt}$ as:

$$
x_{t+dt} = \frac{s(t+dt)}{s(t)} x_t + s(t) s(t+dt) \sigma(t)^2 \left(1 - r^{1 + \alpha}\right) \nabla \log p_t(x_t)
+ s(t+dt) \sigma(t+dt) \sqrt{1 - r^{2 \alpha}} \, z, \quad z \sim \mathcal{N}(0, \mathrm{Id}),

$$

with $r = \sigma(t+dt) / \sigma(t)$ and $\alpha = \alpha(t)$. The noise level of the next state is exactly $\sigma(t+dt)$.

Compared to a Euler-Maruyama step of [`deepinv.sampling.EulerSolver`](https://deepinv.org/api/stubs/deepinv.sampling.EulerSolver.html.md#deepinv.sampling.EulerSolver), it integrates the linear part
and the noise exactly, and freezes the non-linear term. It is thus more accurate when discretizing with few steps.

The ancestral DDPM sampler <sup>[1](#footcite-ho2020denoising)</sup> is obtained for $\alpha = 1$, and the deterministic DDIM sampler
<sup>[2](#footcite-song2020denoising)</sup> for $\alpha = 0$. See [`deepinv.sampling.DDPMSolver`](https://deepinv.org/api/stubs/deepinv.sampling.DDPMSolver.html.md#deepinv.sampling.DDPMSolver) and [`deepinv.sampling.DDIMSolver`](https://deepinv.org/api/stubs/deepinv.sampling.DDIMSolver.html.md#deepinv.sampling.DDIMSolver).

If the parameter $\eta$ of DDIM <sup>[2](#footcite-song2020denoising)</sup> is given, the `alpha` of the SDE is ignored, and replaced on each step by

$$
\alpha_\eta = \frac{\log\left(1 - \eta^2 (1 - r^2)\right)}{2 \log r},

$$

for which the step is exactly the DDIM step with parameter $\eta$.

The `variance` argument sets the variance of the noise added on each step:

- `"small"` (default): use the posterior variance $s(t+dt)^2 \sigma(t+dt)^2 (1 - r^{2 \alpha})$
  ($\tilde{\beta}_t$ of DDPM).
- `"large"`: use the forward variance $s(t)^2 \sigma(t)^2$ ($\beta_t$ for DDPM).
- a float $\zeta \in [0, 1]$: use the variance $\zeta \, s(t+dt)^2 \sigma(t+dt)^2$, i.e. $r^{2 \alpha}$ is replaced on each step by $1 - \zeta$, and the `alpha` of the SDE is ignored. `variance = 0` gives the deterministic DDIM sampler.

#### NOTE
The solver requires `sde.sigma_t`, `sde.scale_t`, `sde.score` and, if `eta` is `None` and `variance` is not a float, `sde.alpha`,
provided by [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE) (and its subclasses) and by [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion).

* **Parameters:**
  * **timesteps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) *,* [*list*](https://docs.python.org/3.12/library/stdtypes.html#list)) – time steps at which the SDE will be discretized.
  * **t_start** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **t_end** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **num_steps** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **eta** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the stochasticity parameter $\eta \in [0, 1]$ of DDIM, optional. If given, it replaces the `alpha` of the SDE by $\alpha_\eta$. If `None` (default), the `alpha` of the SDE is used.
  * **variance** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float)) – the variance of the noise added on each step, either `"small"` (default) for the posterior variance ($\tilde{\beta}_t$ for DDPM),
    `"large"` for the variance of the forward transition ($\beta_t$ for DDPM), or a float $\zeta \in [0, 1]$,
    the fraction of the noise variance renewed on each step, which replaces the `alpha` of the SDE by $\alpha_\zeta$.
    A float cannot be combined with `eta`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – A random number generator for reproducibility.

#### NOTE
You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.

<hr />

* **References:**

* <a id='footcite-ho2020denoising'>**[1]**</a> Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.
* <a id='footcite-song2020denoising'>**[2]**</a> Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *International Conference on Learning Representations*. 2020.

#### step(sde, t0, t1, x0, \*args, \*\*kwargs)

Perform a single ancestral step from time `t0` to time `t1`, with current state `x0`, solving the reverse-time SDE.

* **Parameters:**
  * **sde** ([*deepinv.sampling.EDMDiffusionSDE*](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE) *,* [*deepinv.sampling.PosteriorDiffusion*](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion)) – the SDE to solve, which must provide `sigma_t`, `scale_t`, `score` and, if `eta` is `None` and `variance` is not a float, `alpha`.
  * **t0** ([*float*](https://docs.python.org/3.12/library/functions.html#float) *or* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Time at the start of the step, of size (,).
  * **t1** ([*float*](https://docs.python.org/3.12/library/functions.html#float) *or* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Time at the end of the step, of size (,).
  * **x0** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current state of the system, of size (batch_size, d).
  * **\*args** – additional arguments for the score of the SDE.
  * **\*\*kwargs** – additional keyword arguments for the score of the SDE.
* **Return torch.Tensor, int:**
  Updated state of the system after the step and number of function evaluations (NFE) performed during the step (here 1).
* **Return type:**
  [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [int](https://docs.python.org/3.12/library/functions.html#int)]

<a id="sphx-glr-backref-deepinv-sampling-ancestralsolver"></a>

## Examples using `AncestralSolver`:

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
