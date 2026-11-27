# DDRM

### *class* deepinv.sampling.DDRM(denoiser, schedule='vp', solver='ancestral', alpha=None, num_steps=100, eta=0.85, weight=1.0, eps=1e-6, clip=(0.0, 1.0), verbose=False, dtype=torch.float64, device='cpu', rng=None, \*\*kwargs)

Bases: [`PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion)

Denoising Diffusion Restoration Models (DDRM).

This class implements the Denoising Diffusion Restoration Model (DDRM) described in Kawar *et al.*<sup>[1](#footcite-kawar2022denoising)</sup>.
It inherits from the general posterior sampling framework [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion),
using [`deepinv.sampling.DDRMDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DDRMDataFidelity.html.md#deepinv.sampling.DDRMDataFidelity). It can be coupled with any denoiser, SDE schedule (see [the user guide](https://deepinv.org/user_guide/reconstruction/sampling.html.md#diffusion-sdes)) and solver
(see [the user guide](https://deepinv.org/user_guide/reconstruction/sampling.html.md#sde-ode-solvers)).

DDRM samples from the posterior distribution of a linear inverse problem $y = A x + \epsilon$, with
$\epsilon \sim \mathcal{N}(0, \sigma_y^2 \mathrm{Id})$, using a denoiser and the singular value decomposition (SVD)
$A = U \Sigma V^\top$ of the operator ([`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics)).
In the SVD domain, the measurements decouple into scalar ones, $\bar{y}_i = (U^\top y)_i / s_i$, with noise level
$\nu_i = \sigma_y / s_i$ for each singular value $s_i$. In the VE parametrization $x_t = x_0 + \sigma_t \omega$,
let $\bar{x}_t = V^\top x_t$ and $\bar{x}_0 = V^\top \denoiser{x_t}{\sigma_t}$ be the current state and
the denoised estimate in the SVD domain. A step from $t$ to $t - 1$ samples each component of
$\bar{x}_{t-1} = V^\top x_{t-1}$ as

$$
\bar{x}_{t-1, i} \sim \begin{cases}
\mathcal{N}\left(\bar{x}_{0, i} + \sqrt{1 - \eta^2} \, \sigma_{t-1} \frac{\bar{x}_{t, i} - \bar{x}_{0, i}}{\sigma_t},
\eta^2 \sigma_{t-1}^2\right) & \text{if } s_i = 0, \\
\mathcal{N}\left(\bar{x}_{0, i} + \sqrt{1 - \eta^2} \, \sigma_{t-1} \frac{\bar{y}_i - \bar{x}_{0, i}}{\nu_i},
\eta^2 \sigma_{t-1}^2\right) & \text{if } \sigma_{t-1} < \nu_i, \\
\mathcal{N}\left((1 - \lambda) \bar{x}_{0, i} + \lambda \bar{y}_i, \sigma_{t-1}^2 - \lambda^2 \nu_i^2\right)
& \text{if } \sigma_{t-1} \geq \nu_i,
\end{cases}
$$

where $\eta$ (`eta`) controls the stochasticity and $\lambda$ (`weight`) the weight of the measurements.

The default schedule and solver are chosen to match the original implementaton of Kawar *et al.*<sup>[1](#footcite-kawar2022denoising)</sup>.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser model that can handle different noise levels.
  * **schedule** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*deepinv.sampling.DiffusionSDE*](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE)) – the diffusion SDE (see [the user guide](https://deepinv.org/user_guide/reconstruction/sampling.html.md#diffusion-sdes)),
    either an SDE instance or a string, `"vp"` (default) for [`VariancePreservingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.html.md#deepinv.sampling.VariancePreservingDiffusion),
    `"ve"` for [`VarianceExplodingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VarianceExplodingDiffusion.html.md#deepinv.sampling.VarianceExplodingDiffusion) or `"fm"` for [`FlowMatching`](https://deepinv.org/api/stubs/deepinv.sampling.FlowMatching.html.md#deepinv.sampling.FlowMatching).
    An instance of [`EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE) with custom schedules is also possible. An SDE instance is used as is,
    with its own `alpha`, so that `alpha` is then ignored.
  * **solver** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver of the reverse-time SDE (see [the user guide](https://deepinv.org/user_guide/reconstruction/sampling.html.md#sde-ode-solvers)),
    either a solver instance or a string, e.g. `"euler"` for
    [`deepinv.sampling.EulerSolver`](https://deepinv.org/api/stubs/deepinv.sampling.EulerSolver.html.md#deepinv.sampling.EulerSolver). The default `"ancestral"` is [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver) with `variance = eta**2`,
    and `"ddim"` uses `eta` (stochasticity parameter, see below).
    A solver instance is used as is, with its own time steps and random number generator, so that
    `num_steps` and `rng` are then ignored.
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.12/library/functions.html#float)) – the weight of the noise in the reverse-time SDE, possibly time-dependent, see
    [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE). Only used with a string `schedule`, and only by `"euler"`, `"heun"`
    and by solver instances that use it, such as [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver) without `eta` and with a string `variance`.
    If `None` (default), `1.0` is used.
  * **num_steps** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – the number of time steps of the solver. Default: `100`.
  * **eta** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – stochasticity parameter $\eta \in [0, 1]$ of DDRM, used by the solver and by
    [`deepinv.sampling.DDRMDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DDRMDataFidelity.html.md#deepinv.sampling.DDRMDataFidelity) (default: `0.85`).
  * **weight** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – weight $\lambda$ of the data-fidelity term, denoted $\eta_b$ in
    Kawar *et al.*<sup>[1](#footcite-kawar2022denoising)</sup>. It replaces the deprecated `etab`. Default: `1.0`.
  * **eps** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – small numerical floor for the spectral denominator. Default: `1e-6`.
  * **clip** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – clipping interval for the denoised output, or `None` to disable
    clipping. Default: `(0.0, 1.0)`.
  * **verbose** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `True`, display a progress bar during sampling. Default: `False`.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the solver computations. Default: `torch.float64`.
  * **device** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device for the computations. Default: `"cpu"`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – optional random number generator for reproducibility.

<hr />

* **Examples:**
  DDRM with a pretrained DRUNet denoiser, for the inpainting of a noisy $32 \times 32$ image:
  ```pycon
  >>> import deepinv as dinv
  >>> import torch
  >>> _ = torch.manual_seed(0)
  >>> x = dinv.utils.load_example("butterfly.png", img_size=32)
  >>> physics = dinv.physics.Inpainting(
  ...     img_size=(3, 32, 32), mask=0.5, noise_model=dinv.physics.GaussianNoise(0.05)
  ... )
  >>> y = physics(x)
  >>> denoiser = dinv.models.DRUNet(pretrained="download")
  >>> model = dinv.sampling.DDRM(denoiser=denoiser, num_steps=50)
  >>> xhat = model(y, physics)
  >>> dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x)
  tensor([True])
  >>> dinv.utils.plot([x, y, xhat], titles=["Ground truth", "Measurement", "DDRM"])
  ```

![image](../../_build/plot_directive/api/stubs/deepinv-sampling-DDRM-1.png)

<hr />

* **References:**

* <a id='footcite-kawar2022denoising'>**[1]**</a> Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. *Advances in Neural Information Processing Systems*, 35:23593–23606, 2022.

<a id="sphx-glr-backref-deepinv-sampling-ddrm"></a>

## Examples using `DDRM`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_physics_thumb.png)

[Bring your own physics](https://deepinv.org/auto_examples/basics/demo_custom_physics.html.md)

  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">![](auto_examples/basics/images/thumb/sphx_glr_demo_pretrained_model_thumb.png)

[Use a pretrained model](https://deepinv.org/auto_examples/basics/demo_pretrained_model.html.md)

  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares three diffusion posterior samplers on a noisy inpainting problem:">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_samplers_thumb.png)

[Posterior sampling with DPS, DiffPIR and DDRM](https://deepinv.org/auto_examples/sampling/demo_diffusion_samplers.html.md)

  <div class="sphx-glr-thumbnail-title">Posterior sampling with DPS, DiffPIR and DDRM</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares eight approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.html.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
