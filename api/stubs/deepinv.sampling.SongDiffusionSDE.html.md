# SongDiffusionSDE

### *class* deepinv.sampling.SongDiffusionSDE(beta_t=None, B_t=None, xi_t=None, variance_preserving=False, variance_exploding=False, alpha=0.0, T=1.0, denoiser=None, solver=None, dtype=torch.float64, device=torch.device('cpu'), \*args, \*\*kwargs)

Bases: [`EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE)

Generative diffusion Stochastic Differential Equation.

This class implements the diffusion generative SDE based the formulation from Song *et al.*<sup>[1](#footcite-song2020score)</sup>:

$$
d x_t = -\left(\frac{1}{2} \beta(t) x_t + \frac{1 + \alpha(t)}{2} \xi(t) \nabla \log p_t(x_t) \right) dt + \sqrt{\alpha(t) \xi(t)} d w_t

$$

where $\beta(t)$ is a time-dependent linear drift, $\xi(t)$ is a time-dependent linear diffusion, and
$\alpha(t)$ is weighting the diffusion term.

It corresponds to the reverse-time SDE of the following forward-time SDE:

$$
d x_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\xi(t)} d w_t

$$

Compared to the EDM formulation in [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE), the scale $s(t)$ and noise $\sigma(t)$ schedulers are defined with respect to $\beta(t)$ and $\xi(t)$ as follows:

$$
s(t) = \exp\left(-0.5 \int_0^t \beta(s) ds\right), \quad \sigma(t) = \sqrt{\int_0^t \frac{\xi(s)}{s(s)^2} ds}.

$$

These are the schedules for which the EDM diffusion coefficient $s(t) \sqrt{2 \sigma(t) \sigma'(t)}$ reduces to $\sqrt{\xi(t)}$.
For variance-preserving, it has the closed form $\sigma(t)^2 = 1/s(t)^2 - 1$, which is the DDPM noise level $\sqrt{1 - \bar\alpha(t)} / \sqrt{\bar\alpha(t)}$ for $\bar\alpha = s^2$.

Common choices include the variance-preserving formulation $\beta(t) = \xi(t)$ and the variance-exploding formulation $\beta(t) = 0$.

> - For choosing variance-preserving formulation, set `variance_preserving=True` and `beta_t` and `xi_t` will be automatically set to be the same function.
> - For choosing variance-exploding formulation, set `variance_exploding=True` and `beta_t` will be automatically set to `0`.

#### NOTE
This SDE must be solved going reverse in time i.e. from $t=T$ to $t=0$.

* **Parameters:**
  * **beta_t** (*Callable*) – a time-dependent linear drift of the forward-time SDE.
  * **B_t** (*Callable*) – time integral of beta_t between 0 and t. If None, it is calculated by numerical integration.
  * **xi_t** (*Callable*) – a time-dependent linear diffusion of the forward-time SDE.
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser used to provide an approximation of the score at time $t$: $\nabla \log p_t$.
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function $\alpha(t) = 0$ corresponds to ODE sampling and $\alpha(t) > 0$ corresponds to SDE sampling.
  * **T** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the end time of the forward SDE.
  * **solver** ([*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver for solving the SDE.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the computation, except for the `denoiser` which will use `torch.float32`.
    We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
    most computation cost is from evaluating the `denoiser`, which will be always computed in `torch.float32`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device on which the computation is performed.
  * **\*args** – additional arguments for the [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).
  * **\*\*kwargs** – additional keyword arguments for the [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).

<hr />

* **References:**

* <a id='footcite-song2020score'>**[1]**</a> Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*. 2020.

<a id="sphx-glr-backref-deepinv-sampling-songdiffusionsde"></a>

## Examples using `SongDiffusionSDE`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div>
<!-- thumbnail-parent-div-close --></div>
