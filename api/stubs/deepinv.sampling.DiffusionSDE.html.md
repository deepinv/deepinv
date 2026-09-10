# DiffusionSDE

### *class* deepinv.sampling.DiffusionSDE(forward_drift, forward_diffusion, alpha=1.0, denoiser=None, solver=None, minus_one_one=True, dtype=torch.float64, device=torch.device('cpu'), \*args, \*\*kwargs)

Bases: [`BaseSDE`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.html.md#deepinv.sampling.BaseSDE)

Define the Reverse-time Diffusion Stochastic Differential Equation.

Given a forward-time SDE of the form:

$$
d x_t = f(x_t, t) dt + g(t)d w_t

$$

This class define the following reverse-time SDE:

$$
d x_{t} = \left( f(x_t, t) - \frac{1 + \alpha(t)}{2} g(t)^2 \nabla \log p_t(x_t) \right) dt + g(t) \sqrt{\alpha(t)} d w_{t}.

$$

* **Parameters:**
  * **drift** (*Callable*) – a time-dependent drift function $f(x, t)$ of the forward-time SDE.
  * **diffusion** (*Callable*) – a time-dependent diffusion function $g(t)$ of the forward-time SDE.
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function $\alpha(t) = 0$ corresponds to ODE sampling and $\alpha(t) > 0$ corresponds to SDE sampling.
  * **deepinv.models.Denoiser** – a denoiser used to provide an approximation of the score at time $t$ $\nabla \log p_t$.
  * **solver** ([*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver for solving the SDE.
  * **minus_one_one** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, wrap the denoiser so that SDE states `x` in [-1, 1] are converted to [0, 1] before denoising and mapped back afterward.
    Set `True` for denoisers trained on [0, 1] (all denoisers in [`deepinv.models.Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser));
    set `False` only if the denoiser natively expects [-1, 1].
    This affects only the denoiser interface and usually improves quality when matched to the denoiser’s training range.
    Default: `True`.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the computation, except for the `denoiser` which will use `torch.float32`.
    We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
    most computation cost is from evaluating the `denoiser`, which will be always computed in `torch.float32`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device on which the computation is performed.
  * **\*args** – additional arguments for the [`deepinv.sampling.BaseSDE`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.html.md#deepinv.sampling.BaseSDE).
  * **\*\*kwargs** – additional keyword arguments for the [`deepinv.sampling.BaseSDE`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.html.md#deepinv.sampling.BaseSDE).

#### scale_t(t)

The scale $s(t)$ of the condition distribution $p(x_t \vert x_0) \sim \mathcal{N}(s(t)x_0, s(t)^2 \sigma_t^2 \mathrm{Id})$.

* **Parameters:**
  **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – current time step
* **Returns:**
  the mean of the condition distribution at time step `t`.
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
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### sigma_t(t)

The $\sigma(t)$ of the condition distribution $p(x_t \vert x_0) \sim \mathcal{N}(s(t)x_0, s(t)^2 \sigma_t^2 \mathrm{Id})$.

* **Parameters:**
  **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – current time step
* **Returns:**
  the noise level at time step `t`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-sampling-diffusionsde"></a>

## Examples using `DiffusionSDE`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
