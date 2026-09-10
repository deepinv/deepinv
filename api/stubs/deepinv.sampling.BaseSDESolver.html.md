# BaseSDESolver

### *class* deepinv.sampling.BaseSDESolver(timesteps=None, t_start=None, t_end=None, num_steps=None, rng=None)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class for solving Stochastic Differential Equations (SDEs) from [`deepinv.sampling.BaseSDE`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.html.md#deepinv.sampling.BaseSDE) of the form:

$$
d x_{t} = f(x_t, t) dt + g(t) d w_{t}

$$

where $f$ is the drift term, $g$ is the diffusion coefficient, and $w_t$ is a standard Brownian process.

Currently only supported for fixed time steps for numerical integration.

* **Parameters:**
  * **timesteps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – time steps at which the SDE will be discretized.
  * **t_start** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **t_end** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **num_steps** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – a random number generator for reproducibility, optional.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to display a progress bar during the sampling process, optional. Default to False.

#### NOTE
You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.

#### randn_like(input, seed=None)

Equivalent to [`torch.randn_like()`](https://docs.pytorch.org/docs/stable/generated/torch.randn_like.html#torch.randn_like) but supports a pseudorandom number generator argument.

* **Parameters:**
  * **input** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – The input tensor whose size will be used.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – The seed for the random number generator, if `rng` is provided.
* **Returns:**
  A tensor of the same size as input filled with random numbers from a normal distribution.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

This method uses the `rng` attribute of the class, which is a pseudo-random number generator
for reproducibility. If a seed is provided, it will be used to set the state of `rng` before
generating the random numbers.

#### NOTE
The `rng` attribute must be initialized for this method to work properly.

#### reset_rng()

Reset the random number generator to its initial state.

#### rng_manual_seed(seed=None)

Sets the seed for the random number generator.

* **Parameters:**
  **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed to set for the random number generator. If not provided, the current state of the random number generator is used.
  Note: it will be ignored if the random number generator is not initialized.

#### sample(sde, x_init, seed=None, \*args, timesteps=None, get_trajectory=False, verbose=False, \*\*kwargs)

Solve the Stochastic Differential Equation (SDE) with given time steps.

This function iteratively applies the SDE solver step for each time interval
defined by the provided timesteps.

* **Parameters:**
  * **sde** ([*deepinv.sampling.BaseSDE*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.html.md#deepinv.sampling.BaseSDE)) – the SDE to solve.
  * **x_init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – The initial state of the system.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – The seed for the random number generator, if `rng` is provided.
  * **timesteps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – A sequence of time points at which to solve the SDE. If None, default timesteps will be used.
  * **get_trajectory** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return the full trajectory of the SDE or only the last sample, optional. Default to False.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to display a progress bar during the sampling process, optional. Default to False.
  * **\*args** – Variable length argument list to be passed to the step function.
  * **\*\*kwargs** – Arbitrary keyword arguments to be passed to the step function.
* **Returns:**
  SDEOutput
* **Return type:**
  [SDEOutput](https://deepinv.org/api/stubs/deepinv.sampling.SDEOutput.html.md#deepinv.sampling.SDEOutput)

#### step(sde, t0, t1, x0, \*args, \*\*kwargs)

Perform a single step with step size from time `t0` to time `t1`, with current state `x0`.

* **Parameters:**
  * **sde** ([*deepinv.sampling.BaseSDE*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.html.md#deepinv.sampling.BaseSDE)) – the SDE to solve.
  * **t0** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *or* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Time at the start of the step, of size (,).
  * **t1** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *or* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Time at the end of the step, of size (,).
  * **x0** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current state of the system, of size (batch_size, d).
* **Return torch.Tensor, int:**
  Updated state of the system after the step and number of function evaluations (NFE) performed during the step.
* **Return type:**
  [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [int](https://docs.python.org/3.9/library/functions.html#int)]

<a id="sphx-glr-backref-deepinv-sampling-basesdesolver"></a>

## Examples using `BaseSDESolver`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
