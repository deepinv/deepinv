# HeunSolver

### *class* deepinv.sampling.HeunSolver(timesteps=None, t_start=None, t_end=None, num_steps=None, rng=None)

Bases: [`BaseSDESolver`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.md#deepinv.sampling.BaseSDESolver)

Heun solver for SDEs.

This solver uses the second-order Heun method to numerically integrate SDEs, defined as:

$$
\tilde{x}_{t+dt} &= x_t + f(x_t,t)dt + g(t) W_{dt} \\
x_{t+dt} &= x_t + \frac{1}{2}[f(x_t,t) + f(\tilde{x}_{t+dt},t+dt)]dt + \frac{1}{2}[g(t) + g(t+dt)] W_{dt}

$$

where $W_t$ is a Gaussian random variable with mean 0 and variance dt.

As in Karras *et al.*<sup>[1](#footcite-karras2022elucidating)</sup>, a step that ends at $t = 0$ is an Euler-Maruyama step without
the correction, which would evaluate the score at $t = 0$, where the noise level can vanish (e.g. for
[`deepinv.sampling.VariancePreservingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.md#deepinv.sampling.VariancePreservingDiffusion)).

* **Parameters:**
  * **timesteps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) *,* [*list*](https://docs.python.org/3.12/library/stdtypes.html#list)) – The time steps at which to evaluate the solution.
  * **timesteps** – time steps at which the SDE will be discretized.
  * **t_start** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **t_end** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **num_steps** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – A random number generator for reproducibility.

#### NOTE
You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.

<hr />

* **References:**

* <a id='footcite-karras2022elucidating'>**[1]**</a> Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. *Advances in neural information processing systems*, 35:26565–26577, 2022.

#### step(sde, t0, t1, x0, \*args, \*\*kwargs)

Perform a single Heun step from time `t0` to time `t1`, with current state `x0`:
an Euler-Maruyama prediction, corrected with the drift and diffusion evaluated at the prediction.
The correction is skipped if `t1` is 0.

* **Parameters:**
  * **sde** ([*deepinv.sampling.BaseSDE*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.md#deepinv.sampling.BaseSDE)) – the SDE to solve.
  * **t0** ([*float*](https://docs.python.org/3.12/library/functions.html#float) *or* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Time at the start of the step, of size (,).
  * **t1** ([*float*](https://docs.python.org/3.12/library/functions.html#float) *or* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Time at the end of the step, of size (,).
  * **x0** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current state of the system, of size (batch_size, d).
  * **\*args** – additional arguments for the drift of the SDE.
  * **\*\*kwargs** – additional keyword arguments for the drift of the SDE.
* **Return torch.Tensor, int:**
  Updated state of the system after the step and number of function evaluations (NFE) performed during the step (2, or 1 if `t1` is 0).
* **Return type:**
  [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [int](https://docs.python.org/3.12/library/functions.html#int)]

<a id="sphx-glr-backref-deepinv-sampling-heunsolver"></a>

## Examples using `HeunSolver`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares three diffusion posterior samplers on a noisy inpainting problem:">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_samplers_thumb.png)

[Posterior sampling with DPS, DiffPIR and DDRM](https://deepinv.org/auto_examples/sampling/demo_diffusion_samplers.md)

  <div class="sphx-glr-thumbnail-title">Posterior sampling with DPS, DiffPIR and DDRM</div>
</div>
<!-- thumbnail-parent-div-close --></div>
