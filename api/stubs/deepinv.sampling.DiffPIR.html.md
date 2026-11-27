# DiffPIR

### *class* deepinv.sampling.DiffPIR(denoiser=None, data_fidelity=None, sigma=0.05, max_iter=None, zeta=1.0, weight=7.0, verbose=False, schedule='vp', solver='ancestral', alpha=None, num_steps=100, clip=(0.0, 1.0), dtype=torch.float64, device='cpu', rng=None, \*\*kwargs)

Bases: [`PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion)

Diffusion PnP Image Restoration (DiffPIR).

This class implements the Diffusion PnP image restoration algorithm (DiffPIR) described in Zhu *et al.*<sup>[1](#footcite-zhu2023denoising)</sup>.

The DiffPIR algorithm combines a diffusion reverse process with a half-quadratic splitting (HQS) plug-and-play
proximal data-consistency step. At each diffusion step, the denoised estimate $\denoiser{x_t}{\sigma_t}$
is updated with the proximal operator of the measurement data-fidelity function $f$:

$$
\hat{x}_{0, t} = \operatorname{prox}_{\frac{1}{\rho_t} f}\left(\denoiser{x_t}{\sigma_t}\right),
\qquad \rho_t = \lambda \frac{\sigma_y^2}{\sigma_t^2},
$$

which corresponds to approximating the conditional score via [`deepinv.sampling.DiffPIRDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DiffPIRDataFidelity.html.md#deepinv.sampling.DiffPIRDataFidelity):

$$
\nabla_{x_t} \log p_t(y \vert x_t) \approx
\frac{\hat{x}_{0, t} - \denoiser{x_t}{\sigma_t}}{\sigma_t^2}.
$$

With the default `"ancestral"` solver and `"vp"` schedule, each step is the update of Zhu *et al.*<sup>[1](#footcite-zhu2023denoising)</sup>.
Let $s_t$ and $\sigma_t$ be the scale and noise level of the forward SDE, i.e.
$x_t = s_t \cdot x_0 + s_t \sigma_t \cdot \omega$ with $\omega \sim \mathcal{N}(0, \mathrm{Id})$
(see [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE)). A step of [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver) from $t$
to $t - 1$ is then:

$$
x_{t - 1} = \frac{s_{t-1}}{s_t} x_t + s_t s_{t-1} \sigma_t^2 \left(1 - r^{1 + \alpha}\right) \nabla_{x_t} \log p_t(x_t \vert y)
+ s_{t-1} \sigma_{t-1} \sqrt{1 - r^{2 \alpha}} z,
$$

with $r = \frac{\sigma_{t-1}}{\sigma_t}$ and $z \sim \mathcal{N}(0, \mathrm{Id})$.

The parameter $\zeta$ of DiffPIR is passed as `variance=zeta` to the solver, which sets $r^{2 \alpha} = 1 - \zeta$,
so that a fraction $\zeta$ of the noise variance is renewed on each step.

The conditional score is the sum of the prior score, given by the denoised estimate $\denoiser{x_t / s_t}{\sigma_t}$
through Tweedie’s formula, and of the likelihood term above, divided by $s_t$ by the chain rule since the denoiser
is applied to $x_t / s_t$:

$$
\nabla_{x_t} \log p_t(x_t \vert y)
&= \frac{s_t \denoiser{x_t / s_t}{\sigma_t} - x_t}{s_t^2 \sigma_t^2}
+ \frac{\hat{x}_{0, t} - \denoiser{x_t / s_t}{\sigma_t}}{s_t \sigma_t^2} \\
&= \frac{s_t \hat{x}_{0, t} - x_t}{s_t^2 \sigma_t^2} \\
&= -\frac{\hat{\varepsilon}}{s_t \sigma_t}, \qquad \hat{\varepsilon} = \frac{x_t - s_t \hat{x}_{0, t}}{s_t \sigma_t}.
$$

The data-fidelity term thus replaces the denoised estimate by $\hat{x}_{0, t}$ in Tweedie’s formula, and
$\hat{\varepsilon}$ is the effective noise, such that $x_t = s_t \hat{x}_{0, t} + s_t \sigma_t \hat{\varepsilon}$.
Substituting $x_t$ and the score in the ancestral step gives:

$$
x_{t - 1} &= s_{t-1} \left(\hat{x}_{0, t} + \sigma_t \hat{\varepsilon}\right)
- s_{t-1} \sigma_t \left(1 - \frac{\sigma_{t-1}}{\sigma_t} \sqrt{1 - \zeta}\right) \hat{\varepsilon}
+ s_{t-1} \sigma_{t-1} \sqrt{\zeta} \, z \\
&= s_{t-1} \hat{x}_{0, t} + s_{t-1} \sigma_{t-1} \left(\sqrt{1 - \zeta} \, \hat{\varepsilon} + \sqrt{\zeta} \, z\right).
$$

For [`deepinv.sampling.VariancePreservingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.html.md#deepinv.sampling.VariancePreservingDiffusion), $s_t = \sqrt{\bar{\alpha}_t}$ and
$s_t \sigma_t = \sqrt{1 - \bar{\alpha}_t}$, which gives the sampling step of DiffPIR in Zhu *et al.*<sup>[1](#footcite-zhu2023denoising)</sup>:

$$
\hat{\varepsilon} = \frac{x_t - \sqrt{\bar{\alpha}_t} \hat{x}_{0, t}}{\sqrt{1 - \bar{\alpha}_t}}, \qquad
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_{0, t} + \sqrt{1 - \bar{\alpha}_{t-1}} \left(\sqrt{1 - \zeta} \, \hat{\varepsilon} + \sqrt{\zeta} \, z\right).
$$

The VP schedule is the continuous-time limit of the discrete $\bar{\alpha}_t$ of DDPM, whose noise levels differ by a few percent.
When a different SDE is used, the sampling step uses the same estimate of the conditional score, but $s_t$ and $\sigma_t$ are given by the SDE,
making the sampler more flexible than the original DiffPIR algorithm.

Note that $\sigma_t$ and $\sigma_y$ are both noise levels in $[0, 1]$. The original code of [[212](https://deepinv.org/user_guide/other/biblio.html.md#id68)] uses the noise level in $[-1, 1]$,
therefore the values of $\lambda$ is 4 times smaller than in the original code.

#### NOTE
This method inherits from the general posterior sampling framework [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion),
using [`deepinv.sampling.DiffPIRDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DiffPIRDataFidelity.html.md#deepinv.sampling.DiffPIRDataFidelity). It can be coupled with any denoiser, SDE schedule and solver
(see [the user guide](https://deepinv.org/user_guide/reconstruction/sampling.html.md#sde-ode-solvers)).
The algorithm might require careful tuning of the hyperparameters $\lambda$ (`weight`)
and $\zeta$ (`zeta`) to obtain optimal results.

#### WARNING
In previous versions, the `x_init` argument of [`forward`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion.forward)
was an initial image in $[0, 1]$ (by default $A^{\top} y$), to which the noise of the first step was added.
It is now the noisy initial state of the diffusion itself, as in [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion):
to start from an image, add the noise of the first time step to it.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser network that can handle different noise levels.
    It replaces the deprecated `model`.
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – the measurement data-fidelity operator $f(\cdot, y)$.
    If `None`, defaults to [`deepinv.optim.L2`](https://deepinv.org/api/stubs/deepinv.optim.L2.html.md#deepinv.optim.L2).
  * **sigma** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – fallback measurement noise standard deviation $\sigma_y$ used when
    `physics.noise_model` does not define `sigma`. Default: `0.05`.
  * **max_iter** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – alias for `num_steps` (default: `100`).
  * **zeta** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – fraction $\zeta \in [0, 1]$ of the noise variance renewed on each step of the `"ancestral"`
    solver (its `variance`), controlling the stochasticity of the sampling step. Default: `1.0`.
  * **weight** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – weight $\lambda$ of the data-fidelity step, in $\rho_t = \lambda \sigma_y^2 / \sigma_t^2$
    with both noise levels in the $[0, 1]$ range of the images. The typical values of Zhu *et al.*<sup>[1](#footcite-zhu2023denoising)</sup>,
    between `3.0` and `25.0` depending on the problem.
    It replaces the deprecated `lambda_`. Default: `7.0`.
  * **verbose** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – if `True`, display a progress bar during sampling. Default: `False`.
  * **schedule** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*deepinv.sampling.DiffusionSDE*](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE)) – the diffusion SDE (see [the user guide](https://deepinv.org/user_guide/reconstruction/sampling.html.md#diffusion-sdes)),
    either an SDE instance or a string, `"vp"` (default) for [`deepinv.sampling.VariancePreservingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.html.md#deepinv.sampling.VariancePreservingDiffusion)
    or `"ve"` for [`deepinv.sampling.VarianceExplodingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VarianceExplodingDiffusion.html.md#deepinv.sampling.VarianceExplodingDiffusion). An SDE instance, e.g. with a custom noise
    level `sigma_t`, is used as is, with its own `alpha`, so that `alpha` is then ignored.
  * **solver** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver of the reverse-time SDE (see [the user guide](https://deepinv.org/user_guide/reconstruction/sampling.html.md#sde-ode-solvers)),
    either a solver instance or the name of a solver class in lowercase without `Solver`, e.g. `"euler"` for
    [`deepinv.sampling.EulerSolver`](https://deepinv.org/api/stubs/deepinv.sampling.EulerSolver.html.md#deepinv.sampling.EulerSolver). The default `"ancestral"` is [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver) with `variance = zeta`.
    A solver instance is used as is, with its own time steps and random number generator, so that
    `num_steps` and `rng` are then ignored.
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.12/library/functions.html#float)) – weight of the noise in the reverse-time SDE, see [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).
    Only used with a string `schedule`, and only by `"euler"`, `"heun"` and by solver instances that use it,
    such as [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver) without `eta` and with a string `variance`.
    If `None` (default), `1.0` is used.
  * **num_steps** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – the number of time steps of the solver (default: `100`).
  * **clip** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – clipping interval for the denoised output before the proximal step, or `None` to
    disable clipping. Default: `(0.0, 1.0)`.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the solver computations. Default: `torch.float64`.
  * **device** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – the device to use for the computations. Default: `"cpu"`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – optional random number generator for reproducibility.

<hr />

* **Examples:**
  DiffPIR with a pretrained DRUNet denoiser, for the inpainting of a noisy $32 \times 32$ image:
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
  >>> model = dinv.sampling.DiffPIR(denoiser=denoiser, num_steps=50)
  >>> xhat = model(y, physics)
  >>> dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x)
  tensor([True])
  >>> dinv.utils.plot([x, y, xhat], titles=["Ground truth", "Measurement", "DiffPIR"])
  ```

![image](../../_build/plot_directive/api/stubs/deepinv-sampling-DiffPIR-1.png)

<hr />

* **References:**

* <a id='footcite-zhu2023denoising'>**[1]**</a> Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 1219–1229. 2023.

<a id="sphx-glr-backref-deepinv-sampling-diffpir"></a>

## Examples using `DiffPIR`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares three diffusion posterior samplers on a noisy inpainting problem:">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_samplers_thumb.png)

[Posterior sampling with DPS, DiffPIR and DDRM](https://deepinv.org/auto_examples/sampling/demo_diffusion_samplers.html.md)

  <div class="sphx-glr-thumbnail-title">Posterior sampling with DPS, DiffPIR and DDRM</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares eight approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.html.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
