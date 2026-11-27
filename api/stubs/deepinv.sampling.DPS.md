# DPS

### *class* deepinv.sampling.DPS(denoiser, schedule='vp', solver='ancestral', alpha=None, num_steps=1000, weight=1.0, guidance='norm', verbose=False, device='cpu', dtype=torch.float64, rng=None, \*\*kwargs)

Bases: [`PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.md#deepinv.sampling.PosteriorDiffusion)

Diffusion Posterior Sampling (DPS).

This class implements the Diffusion Posterior Sampling algorithm (DPS) described in Chung *et al.*<sup>[1](#footcite-chung2022diffusion)</sup>.

DPS samples from the posterior distribution of an inverse problem $y = A(x) + \epsilon$, with
$\epsilon \sim \mathcal{N}(0, \sigma_y^2 \mathrm{Id})$ and $A$ can be nonlinear. It solves the reverse-time SDE given by `schedule`
with `solver`, where the score of the prior is replaced by the conditional score, given by Bayes’ rule:

$$
\nabla_{x_t} \log p_t(x_t \vert y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y \vert x_t).
$$

The prior score is given by the denoiser $\denoiser{\cdot}{\sigma}$ through Tweedie’s formula.

The likelihood $p_t(y \vert x_t)$ is intractable.
In the VE parametrerization, Chung *et al.*<sup>[1](#footcite-chung2022diffusion)</sup> estimates its score by:

$$
\nabla_{x_t} \log p_t(y \vert x_t) \approx -\lambda \nabla_{x_t} \|y - A(\denoiser{x_t}{\sigma_t})\|,
$$

where $\lambda$ is the `weight` of the data-fidelity term. With `guidance="annealed"`, it is the score of a
Gaussian likelihood whose variance is increased by $\sigma_t^2$ to account for the error of the denoised estimate:

$$
\nabla_{x_t} \log p_t(y \vert x_t) \approx
-\lambda \nabla_{x_t} \frac{\|y - A(\denoiser{x_t}{\sigma_t})\|^2}{2 (\sigma_y^2 + \sigma_t^2)},
$$

which puts $\lambda$ on the same scale as the other noisy data-fidelity terms, see [`deepinv.sampling.DPSDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DPSDataFidelity.md#deepinv.sampling.DPSDataFidelity).
Both gradients are computed by backpropagation through the denoiser, so that each step of DPS costs a backward pass
on top of the forward pass of the denoiser, unlike [`deepinv.sampling.DiffPIR`](https://deepinv.org/api/stubs/deepinv.sampling.DiffPIR.md#deepinv.sampling.DiffPIR) and [`deepinv.sampling.DDRM`](https://deepinv.org/api/stubs/deepinv.sampling.DDRM.md#deepinv.sampling.DDRM).

#### NOTE
This method inherits from the general posterior sampling framework [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.md#deepinv.sampling.PosteriorDiffusion), using the DPS data fidelity [`deepinv.sampling.DPSDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DPSDataFidelity.md#deepinv.sampling.DPSDataFidelity).
It can be coupled with any denoiser, SDE and solver, which allows for a wide range of sampling algorithms.
The default schedule and solver correspond to the original DPS algorithm in Chung *et al.* [[30](https://deepinv.org/user_guide/other/biblio.md#id66)].

See [Posterior sampling with DPS, DiffPIR and DDRM](https://deepinv.org/auto_examples/sampling/demo_diffusion_samplers.md#sphx-glr-auto-examples-sampling-demo-diffusion-samplers-py) for a comparison with DiffPIR and DDRM, and
[Building your diffusion posterior sampling method using SDEs](https://deepinv.org/auto_examples/sampling/demo_diffusion_sde.md#sphx-glr-auto-examples-sampling-demo-diffusion-sde-py) for more examples on diffusion-based sampling methods.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.md#deepinv.models.Denoiser)) – a denoiser network that can handle different noise levels
  * **schedule** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*deepinv.sampling.DiffusionSDE*](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.md#deepinv.sampling.DiffusionSDE)) – the diffusion SDE (see [the user guide](https://deepinv.org/user_guide/reconstruction/sampling.md#diffusion-sdes)), either an SDE instance or a string,
    `"vp"` (default, which matches the original implementation) for [`deepinv.sampling.VariancePreservingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.md#deepinv.sampling.VariancePreservingDiffusion)
    or `"ve"` for [`deepinv.sampling.VarianceExplodingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VarianceExplodingDiffusion.md#deepinv.sampling.VarianceExplodingDiffusion).
    An SDE instance, e.g. with a custom noise level `sigma_t`, is used as is, with its own `alpha`, so that `alpha` is then ignored.
  * **solver** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.md#deepinv.sampling.BaseSDESolver)) – 

    the solver of the reverse-time SDE, either a solver instance, or one of:
    - `"ancestral"` (default) for [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.md#deepinv.sampling.AncestralSolver), which gives the DDPM sampler for `alpha=1` and the DDIM sampler for `alpha=0`,
    - `"ddpm"` for [`deepinv.sampling.DDPMSolver`](https://deepinv.org/api/stubs/deepinv.sampling.DDPMSolver.md#deepinv.sampling.DDPMSolver), the sampler of the original implementation,
    - `"ddim"` for the deterministic [`deepinv.sampling.DDIMSolver`](https://deepinv.org/api/stubs/deepinv.sampling.DDIMSolver.md#deepinv.sampling.DDIMSolver),
    - `"euler"` for [`deepinv.sampling.EulerSolver`](https://deepinv.org/api/stubs/deepinv.sampling.EulerSolver.md#deepinv.sampling.EulerSolver),
    - `"heun"` for [`deepinv.sampling.HeunSolver`](https://deepinv.org/api/stubs/deepinv.sampling.HeunSolver.md#deepinv.sampling.HeunSolver).

    The `alpha` of the SDE is ignored by `"ddpm"` and `"ddim"`. A solver instance is used as is, with its own time steps and random number generator,
    so that `num_steps` and `rng` are then ignored.
  * **num_steps** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – the number of time steps of the solver (default: 1000)
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.12/library/functions.html#float)) – the weight of the noise in the reverse-time SDE, possibly time-dependent, see [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.md#deepinv.sampling.DiffusionSDE).
    Only used with a string `schedule`. If `None` (default), `1.0` is used, which corresponds to the original DDPM sampling scheme.
    Setting it to 0 corresponds to the deterministic DDIM sampling scheme.
    Intermediate values differ from the parameter $\eta$ of DDIM, see [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.md#deepinv.sampling.AncestralSolver) for the exact relation.
  * **weight** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the weight of the data fidelity term in the approximation of the likelihood gradient. Default to 1.0.
  * **guidance** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – the form of the guidance, passed to [`deepinv.sampling.DPSDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DPSDataFidelity.md#deepinv.sampling.DPSDataFidelity).
    `"norm"` (default) differentiates the residual norm, as in the original paper; `"annealed"` differentiates
    the Gaussian negative log-likelihood with the annealed variance $\sigma_y^2 + \sigma_t^2$, which puts
    `weight` on the same scale as the other noisy data-fidelity terms.
  * **verbose** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `True`, print the progress of the algorithm
  * **device** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – the device to use for the computations

#### TIP
For few steps sampling (e.g. `num_steps < 50`), the ancestral solvers (`"ancestral"`, `"ddpm"` or `"ddim"`) are recommended over `"euler"` and `"heun"`.
For many steps sampling, all solvers give similar results.

<hr />

* **Examples:**
  DPS with a pretrained DRUNet denoiser, for the inpainting of a noisy $32 \times 32$ image:
  ```pycon
  >>> import deepinv as dinv
  >>> import torch
  >>> _ = torch.manual_seed(0)
  >>> x = dinv.utils.load_example("butterfly.png", img_size=32)
  >>> physics = dinv.physics.Inpainting(
  ...     img_size=(3, 32, 32), mask=0.7, noise_model=dinv.physics.GaussianNoise(0.05)
  ... )
  >>> y = physics(x)
  >>> denoiser = dinv.models.DRUNet(pretrained="download")
  >>> model = dinv.sampling.DPS(denoiser=denoiser, num_steps=50, weight=5.0, guidance="annealed")
  >>> xhat = model(y, physics)
  >>> dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x)
  tensor([True])
  >>> dinv.utils.plot([x, y, xhat], titles=["Ground truth", "Measurement", "DPS"])
  ```

![image](../../_build/plot_directive/api/stubs/deepinv-sampling-DPS-1.png)

<hr />

* **References:**

* <a id='footcite-chung2022diffusion'>**[1]**</a> Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In *The Eleventh International Conference on Learning Representations*. 2022.

<a id="sphx-glr-backref-deepinv-sampling-dps"></a>

## Examples using `DPS`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares three diffusion posterior samplers on a noisy inpainting problem:">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_samplers_thumb.png)

[Posterior sampling with DPS, DiffPIR and DDRM](https://deepinv.org/auto_examples/sampling/demo_diffusion_samplers.md)

  <div class="sphx-glr-thumbnail-title">Posterior sampling with DPS, DiffPIR and DDRM</div>
</div>
<!-- thumbnail-parent-div-close --></div>
