# EDMDiffusionSDE

### *class* deepinv.sampling.EDMDiffusionSDE(sigma_t, scale_t=None, sigma_prime_t=None, scale_prime_t=None, variance_preserving=False, variance_exploding=False, alpha=1.0, T=1.0, denoiser=None, solver=None, dtype=torch.float64, device=torch.device('cpu'), \*args, \*\*kwargs)

Bases: [`DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE)

Generative diffusion Stochastic Differential Equation.

This class implements the diffusion generative SDE based on the formulation from Karras *et al.*<sup>[1](#footcite-karras2022elucidating)</sup> (with $\beta(t) = \alpha(t) s(t)^2 \sigma(t) \sigma'(t)$):

$$
d x_t = \left(\frac{s'(t)}{s(t)} x_t - (1 + \alpha(t)) s(t)^2 \sigma(t) \sigma'(t) \nabla \log p_t(x_t) \right) dt + s(t) \sqrt{2 \alpha(t) \sigma(t) \sigma'(t)} d w_t

$$

where $s(t)$ is a time-dependent scale, $\sigma(t)$ is a time-dependent noise level, and $\alpha(t)$ is weighting the diffusion term.
It corresponds to the reverse-time SDE of the following forward-time SDE:

$$
d x_t = \frac{s'(t)}{s(t)} x_t dt + s(t) \sqrt{2 \sigma(t) \sigma'(t)} d w_t

$$

The scale $s(t)$ and noise $\sigma(t)$ schedulers must satisfy $s(0) = 1$, $\sigma(0) = 0$ and $\lim_{t \to \infty} \sigma(t) = +\infty$.

Common choices include the variance-preserving formulation $s(t) = \left(1 + \sigma(t)^2\right)^{-1/2}$ and the variance-exploding formulation $s(t) = 1$.

> - For choosing variance-preserving formulation, set `variance_preserving=True` and do not provide `scale_t` and `scale_prime_t`.
> - For choosing variance-exploding formulation, set `variance_exploding=True` and do not provide `scale_t` and `scale_prime_t`.

#### NOTE
This SDE must be solved by going reverse in time i.e. from $t=T$ to $t=0$.

* **Parameters:**
  * **sigma_t** (*Callable*) – a time-dependent noise level schedule.
    It takes a time step `t` (either a Python `float` or a `torch.Tensor`) as input  and returns the noise level at time `t` (either a Python `float` or a `torch.Tensor`).
    Note that this is a required argument.
  * **scale_t** (*Callable*) – a time-dependent scale schedule.
    It takes a time step `t` (either a Python `float` or a `torch.Tensor`) as input  and returns the noise level at time `t` (either a Python `float` or a `torch.Tensor`).
    If not provided, it will be set to $s(t) = \left(1 + \sigma(t)^2\right)^{-1/2}$ if `variance_preserving=True`, or $s(t) = 1$ if `variance_exploding=True`.
    If both `variance_preserving` and `variance_exploding` are `False`, `scale_t` must be provided. Default to `None`.
  * **sigma_prime_t** (*Callable*) – the derivative of `sigma_t`.
    It takes a time step `t` (either a Python `float` or a `torch.Tensor`) as input and returns the noise level at time `t` (either a Python `float` or a `torch.Tensor`).
    If not provided, it will be computed using autograd. Default to `None`.
  * **scale_prime_t** (*Callable*) – the derivative of `scale_t`.
    It takes a time step `t` (either a Python `float` or a `torch.Tensor`) as input and returns the noise level at time `t` (either a Python `float` or a `torch.Tensor`).
    If not provided, it will be computed using autograd. Default to `None`.
  * **variance_preserving** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use a variance-preserving diffusion schedule, which imposes $s(t) = \left(1 + \sigma(t)^2\right)^{-1/2}$. Default to `False`.
  * **variance_exploding** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use a variance-exploding diffusion schedule, which imposes $s(t) = 1$. Default to `False`.
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function $\alpha(t) = 0$ corresponds to ODE sampling and $\alpha(t) > 0$ corresponds to SDE sampling.
  * **T** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the end time of the forward SDE. Default to `1.0`.
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser used to provide an approximation of the score at time $t$: $\nabla \log p_t$. Default to `None`.
  * **solver** ([*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver for solving the SDE. Default to `None`.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the computation, except for the `denoiser` which will use `torch.float32`.
    We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
    most computation cost is from evaluating the `denoiser`, which will be always computed in `torch.float32`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device on which the computation is performed. Default to CPU.
  * **\*args** – additional arguments for the [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).
  * **\*\*kwargs** – additional keyword arguments for the [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).

<hr />

* **References:**

* <a id='footcite-karras2022elucidating'>**[1]**</a> Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. *Advances in neural information processing systems*, 35:26565–26577, 2022.

#### sample_init(shape, rng=None, t=None)

Sample from the initial distribution of the reverse-time diffusion SDE, which is a Gaussian with zero mean and covariance matrix :math:\` s(t)^2 sigma(t)^2 operatorname{Id}\`.

#### NOTE
The state is drawn at the time the solver starts from, which is `timestep[0]` of the solver, which is not necessarily the end time $T$ of the forward SDE.
The `timesteps` of the solver must be decreasing.

* **Parameters:**
  * **shape** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – The shape of the sample to generate
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – Random number generator for reproducibility
  * **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – the time at which the state is drawn. If `None`, defaults to end time `T`
* **Returns:**
  A sample from the prior distribution
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### score(x, t, \*args, \*\*kwargs)

Approximating the score function $\nabla \log p_t$ by the denoiser.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – current state
  * **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – current time step
  * **\*args** – additional arguments for the `denoiser`.
  * **\*\*kwargs** – additional keyword arguments for the `denoiser`, e.g., `class_labels` for class-conditional models.
* **Returns:**
  the score function $\nabla \log p_t(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-sampling-edmdiffusionsde"></a>

## Examples using `EDMDiffusionSDE`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
