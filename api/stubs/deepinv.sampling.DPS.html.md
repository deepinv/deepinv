# DPS

### *class* deepinv.sampling.DPS(denoiser, schedule='vp', solver='ddpm', alpha=1.0, num_steps=1000, weight=1.0, guidance='norm', verbose=False, device='cpu', dtype=torch.float64, rng=None, \*\*kwargs)

Bases: [`PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion)

Diffusion Posterior Sampling (DPS).

This class implements the Diffusion Posterior Sampling algorithm (DPS) described in Chung *et al.*<sup>[1](#footcite-chung2022diffusion)</sup>.

DPS is an approximation of a gradient-based posterior sampling algorithm,
which has minimal assumptions on the forward model. The only restriction is that
the measurement model has to be differentiable, which is generally the case.

The algorithm solves the reverse-time SDE specified by the `schedule` argument, using the solver specified by the `solver` argument, and approximating the conditional score by the DPS data fidelity term, which is defined as follows:

$$
\nabla_{x_t} \log p_t(y|x_t) \approx -\lambda \nabla_{x_t} \|y - A D_{\sigma_t}(x_t)\|
$$

where $\denoiser{\cdot}{\sigma}$ is a denoising network for noise level $\sigma$, and $\lambda$ is a hyperparameter that controls the weight of the data fidelity term in the approximation of the likelihood gradient.

#### NOTE
This method inherits from the general posterior sampling framework [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion), using the DPS data fidelity [`deepinv.sampling.DPSDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DPSDataFidelity.html.md#deepinv.sampling.DPSDataFidelity).
It can be coupled with any denoiser, SDE and solver, which allows for a wide range of sampling algorithms.
The default parameters correspond to the original DPS algorithm in Chung *et al.* [[29](https://deepinv.org/user_guide/other/biblio.html.md#id65)].

Please refer to the example [Building your diffusion posterior sampling method using SDEs](https://deepinv.org/auto_examples/sampling/demo_diffusion_sde.html.md#sphx-glr-auto-examples-sampling-demo-diffusion-sde-py) for more examples on diffusion-based sampling methods.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser network that can handle different noise levels
  * **schedule** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – the noise schedule to use, either `"vp"` (default, which matches the original implementation) for the variance preserving noise schedule, or `"ve"` for the variance exploding noise schedule.
  * **solver** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – 

    the solver of the reverse-time SDE, either a solver instance, or one of:
    - `"ancestral"` (default) for [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver), which gives the DDPM sampler for `alpha=1` and the DDIM sampler for `alpha=0`,
    - `"ddpm"` for [`deepinv.sampling.DDPMSolver`](https://deepinv.org/api/stubs/deepinv.sampling.DDPMSolver.html.md#deepinv.sampling.DDPMSolver), the sampler of the original implementation,
    - `"ddim"` for the deterministic [`deepinv.sampling.DDIMSolver`](https://deepinv.org/api/stubs/deepinv.sampling.DDIMSolver.html.md#deepinv.sampling.DDIMSolver),
    - `"euler"` for [`deepinv.sampling.EulerSolver`](https://deepinv.org/api/stubs/deepinv.sampling.EulerSolver.html.md#deepinv.sampling.EulerSolver).

    The `alpha` of the SDE is ignored by `"ddpm"` and `"ddim"`. A solver instance is used as is, with its own time steps and random number generator,
    so that `num_steps` and `rng` are then ignored.
  * **num_steps** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – the number of time steps of the solver (default: 1000)
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.12/library/functions.html#float)) – the weight of the noise in the reverse-time SDE, possibly time-dependent, see [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).
    Default to 1.0, which corresponds to the original DDPM sampling scheme. Setting it to 0 corresponds to the deterministic DDIM sampling scheme.
    Intermediate values differ from the parameter $\eta$ of DDIM, see [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver) for the exact relation.
  * **weight** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the weight of the data fidelity term in the approximation of the likelihood gradient. Default to 1.0.
  * **guidance** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – the form of the guidance, passed to [`deepinv.sampling.DPSDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DPSDataFidelity.html.md#deepinv.sampling.DPSDataFidelity).
    `"norm"` (default) differentiates the residual norm, as in the original paper; `"annealed"` differentiates
    the Gaussian negative log-likelihood with the annealed variance $\sigma_y^2 + \sigma_t^2$, which puts
    `weight` on the same scale as the other noisy data-fidelity terms.
  * **verbose** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `True`, print the progress of the algorithm
  * **device** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – the device to use for the computations

#### TIP
For few steps sampling (e.g. `num_steps < 50`), the ancestral solvers (`"ancestral"`, `"ddpm"` or `"ddim"`) are recommended over `"euler"`.
For many steps sampling, all solvers give similar results.

<hr />

* **References:**

* <a id='footcite-chung2022diffusion'>**[1]**</a> Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In *The Eleventh International Conference on Learning Representations*. 2022.

<a id="sphx-glr-backref-deepinv-sampling-dps"></a>

## Examples using `DPS`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in chung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_dps_thumb.png)

[DPS – Posterior Sampling with Diffusion Models](https://deepinv.org/auto_examples/sampling/demo_dps.html.md)

  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div>
<!-- thumbnail-parent-div-close --></div>
