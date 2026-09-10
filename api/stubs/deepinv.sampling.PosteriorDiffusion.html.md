# PosteriorDiffusion

### *class* deepinv.sampling.PosteriorDiffusion(data_fidelity=None, denoiser=None, sde=None, solver=None, dtype=torch.float64, device=torch.device('cpu'), verbose=False, minus_one_one=True, \*args, \*\*kwargs)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Posterior distribution sampling  for inverse problems using diffusion models by Reverse-time Stochastic Differential Equation (SDE).

Consider the acquisition model:

$$
y = \noise{\forw{x}}.

$$

This class defines the reverse-time SDE for the posterior distribution $p(x|y)$ given the data $y$:

$$
d\, x_t = \left( f(x_t, t) - \frac{1 + \alpha(t)}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t | y) \right) d\,t + g(t) \sqrt{\alpha(t)} d\, w_{t}

$$

where $f$ is the drift term, $g$ is the diffusion coefficient and $w$ is the standard Brownian motion. The drift term and the diffusion coefficient are defined by the underlying (unconditional) forward-time SDE `sde`. The (conditional) score function $\nabla_{x_t} \log p_t(x_t | y)$ can be decomposed using the Bayes’ rule:

$$
\nabla_{x_t} \log p_t(x_t | y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y | x_t).

$$

The first term is the score function of the unconditional SDE, which is typically approximated by a MMSE denoiser using the well-known Tweedie’s formula, while the second term is approximated by the (noisy) data-fidelity term. We implement various data-fidelity terms in [`deepinv.sampling.NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.html.md#deepinv.sampling.NoisyDataFidelity).

* **Parameters:**
  * **data_fidelity** ([*deepinv.sampling.NoisyDataFidelity*](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.html.md#deepinv.sampling.NoisyDataFidelity)) – the noisy data-fidelity term, used to approximate the score $\nabla_{x_t} \log p_t(y \vert x_t)$. Default to [`deepinv.optim.ZeroFidelity`](https://deepinv.org/api/stubs/deepinv.optim.ZeroFidelity.html.md#deepinv.optim.ZeroFidelity), which corresponds to the zero data-fidelity term and the sampling process boils down to the unconditional SDE sampling.
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser used to provide an approximation of the (unconditional) score at time $t$ $\nabla \log p_t$.
  * **sde** ([*deepinv.sampling.DiffusionSDE*](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE)) – the forward-time SDE, which defines the drift and diffusion terms of the reverse-time SDE.
  * **solver** ([*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver for the SDE. If not specified, the solver from the `sde` will be used.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – the data type of the sampling solver, except for the `denoiser` which will use `torch.float32`.
    We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since most computation cost is from evaluating the `denoiser`, which will be always computed in `torch.float32`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – the device for the computations.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to display a progress bar during the sampling process, optional. Default to `False`.
  * **minus_one_one** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – 

    If `True`, wrap the denoiser so that SDE states `x` in `[-1, 1]` are converted to `[0, 1]` before denoising and mapped back afterward.
    - Set `True` for denoisers trained on `[0, 1]` data range (all denoisers in [`deepinv.models.Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)).
    - Set `False` only if the denoiser natively expects `[-1, 1]` data range.

    This affects only the denoiser interface and usually improves quality when matched to the denoiser’s training range.
    Default: `True`.

#### forward(y, physics, x_init=None, seed=None, timesteps=None, denoise_output=True, get_trajectory=False, \*args, \*\*kwargs)

Sample the posterior distribution $p(x|y)$ given the data measurement $y$.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the data measurement.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – the forward operator.
  * **x_init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – the initial value for the sampling, can be a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) or a tuple `(B, C, H, W)`, indicating the shape of the initial point, matching the shape of `physics` and `y`. In this case, the initial value is taken randomly following the distribution of the `sde` at the first time step of the solver.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the random seed for reproducibility, the same samples will be generated for the same seed. Default to `None`.
  * **timesteps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the time steps for the solver. If `None`, the default time steps in the solver will be used. Default to `None`.
  * **denoise_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to perform an additional denoising step at the end of the sampling process, which can improve the quality of the generated samples. Default to `True`.
  * **get_trajectory** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return the full trajectory of the SDE or only the last sample, optional. Default to `False`.
  * **\*args** – the additional arguments for the solver.
  * **\*\*kwargs** – the additional keyword arguments for the solver.
* **Returns:**
  the generated sample ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `(B, C, H, W)`) if `get_trajectory` is `False`. Otherwise, returns a tuple ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) of shape `(B, C, H, W)` and `(N, B, C, H, W)` where `N` is the number of steps.

#### score(y, physics, x, t, \*args, \*\*kwargs)

Approximating the conditional score $\nabla_{x_t} \log p_t(x_t \vert y)$.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the data measurement.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – the forward operator.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the current state.
  * **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – the current time step.
  * **\*args** – additional arguments for the score function of the unconditional SDE.
  * **\*\*kwargs** – additional keyword arguments for the score function of the unconditional SDE.
* **Returns:**
  the score function $\nabla_{x_t} \log p_t(x_t \vert y)$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-sampling-posteriordiffusion"></a>

## Examples using `PosteriorDiffusion`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
