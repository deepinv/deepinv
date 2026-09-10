# BaseSDE

### *class* deepinv.sampling.BaseSDE(drift, diffusion, solver=None, dtype=torch.float64, device=torch.device('cpu'), \*args, \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class for Stochastic Differential Equation (SDE):

$$
d x_{t} = f(x_t, t) dt + g(t) d w_{t}

$$

where $f$ is the drift term, $g$ is the diffusion coefficient and $w$ is the standard Brownian motion.
It defines the common interface for drift and diffusion functions.

* **Parameters:**
  * **drift** (*Callable*) – a time-dependent drift function $f(x, t)$
  * **diffusion** (*Callable*) – a time-dependent diffusion function $g(t)$
  * **solver** ([*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver for solving the SDE.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – the data type of the computations.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – the device for the computations.

#### discretize(x, t, \*args, \*\*kwargs)

Discretize the SDE at the given time step.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – current state.
  * **t** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – discretized time step.
  * **\*args** – additional arguments for the drift.
  * **\*\*kwargs** – additional keyword arguments for the drift.
* **Return tuple[torch.Tensor, torch.Tensor]:**
  discretized drift and diffusion.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### forward(x_init=None, seed=None, get_trajectory=False, \*args, \*\*kwargs)

The forward function corresponds to SDE sampling.

* **Parameters:**
  * **x_init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – initial value.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the pseudo-random number generator used in the solver.
  * **get_trajectory** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return the full trajectory of the SDE or only the last sample, optional. Default to False
  * **\*args** – additional arguments for the solver.
  * **\*\*kwargs** – additional keyword arguments for the solver.
* **Returns:**
  the generated sample ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `(B, C, H, W)`) if `get_trajectory` is `False`. Otherwise, returns ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) of shape `(B, C, H, W)` and `(N, B, C, H, W)` where `N` is the number of steps.
* **Return type:**
  [*SDEOutput*](https://deepinv.org/api/stubs/deepinv.sampling.SDEOutput.html.md#deepinv.sampling.SDEOutput)

#### sample(x_init=None, seed=None, get_trajectory=False, \*args, \*\*kwargs)

Solve the SDE with the given timesteps.

* **Parameters:**
  * **x_init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – initial value.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the pseudo-random number generator used in the solver.
  * **get_trajectory** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return the full trajectory of the SDE or only the last sample, optional. Default to False
  * **\*args** – additional arguments for the solver.
  * **\*\*kwargs** – additional keyword arguments for the solver.

:return : the generated sample ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `(B, C, H, W)`) if `get_trajectory` is `False`. Otherwise, returns ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) of shape `(B, C, H, W)` and `(N, B, C, H, W)` where `N` is the number of steps.

#### sample_init(shape, rng=None, t=None)

Sample from the initial distribution of the SDE.

* **Parameters:**
  * **shape** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *|* [*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) – The shape of the the sample, of the form `(B, C, H, W)`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – Random number generator for reproducibility.
  * **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – the time at which the state is drawn. If `None`, defaults to the end time `T`.

<a id="sphx-glr-backref-deepinv-sampling-basesde"></a>

## Examples using `BaseSDE`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
