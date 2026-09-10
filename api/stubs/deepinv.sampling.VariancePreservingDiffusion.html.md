# VariancePreservingDiffusion

### *class* deepinv.sampling.VariancePreservingDiffusion(denoiser=None, beta_min=0.1, beta_max=20.0, alpha=0.0, T=1.0, scaled_linear=False, solver=None, dtype=torch.float64, device=torch.device('cpu'), \*args, \*\*kwargs)

Bases: [`SongDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.SongDiffusionSDE.html.md#deepinv.sampling.SongDiffusionSDE)

Variance-Preserving Stochastic Differential Equation (VP-SDE).

This class implements the reverse-time SDE of the Variance-Preserving SDE (VP-SDE) Song *et al.*<sup>[1](#footcite-song2020score)</sup>.

The forward-time SDE is defined as follows:

$$
d x_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\beta(t)} d w_t \quad \mbox{ where } \beta(t) = \beta_{\mathrm{min}}  + t \left( \beta_{\mathrm{max}} - \beta_{\mathrm{min}} \right)

$$

The reverse-time SDE is defined as follows:

$$
d x_t = -\left(\frac{1}{2} \beta(t) x_t + \frac{1 + \alpha(t)}{2} \beta(t) \nabla \log p_t(x_t) \right) dt + \sqrt{\alpha(t) \beta(t)} d w_t

$$

where $\alpha(t)$ is weighting the diffusion term.

This class is the reverse-time SDE of the VP-SDE, serving as the generation process.

#### NOTE
This SDE must be solved going reverse in time i.e. from $t=T$ to $t=0$.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser used to provide an approximation of the score at time $t$: $\nabla \log p_t$.
  * **beta_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the minimum noise level.
  * **beta_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the maximum noise level.
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function $\alpha(t) = 0$ corresponds to ODE sampling and $\alpha(t) > 0$ corresponds to SDE sampling.
  * **T** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the end time of the forward SDE. Default to `1.0`.
  * **scaled_linear** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use the scaled linear beta schedule. If `False`, uses the more standard linear schedule. Default to `False`.
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

<a id="sphx-glr-backref-deepinv-sampling-variancepreservingdiffusion"></a>

## Examples using `VariancePreservingDiffusion`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div>
<!-- thumbnail-parent-div-close --></div>
