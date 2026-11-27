# DDPMSolver

### *class* deepinv.sampling.DDPMSolver(timesteps=None, t_start=None, t_end=None, num_steps=None, variance='small', rng=None)

Bases: [`AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver)

DDPM solver for reverse-time diffusion SDEs.

Ancestral sampler of DDPM <sup>[1](#footcite-ho2020denoising)</sup>, i.e. [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver) with $\eta = 1$ and the stochastic term `alpha` of the SDE is ignored (fixed to 1).

For [`deepinv.sampling.VariancePreservingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.html.md#deepinv.sampling.VariancePreservingDiffusion) with time steps matching the training time steps of a discrete DDPM model,
this is exactly the DDPM sampler with the posterior variance $\tilde{\beta}_t$.
With `variance="large"`, the posterior variance $\tilde{\beta}_t$ is replaced by the variance $\beta_t$ of the forward transition,
see Section 3.2 of Ho *et al.*<sup>[1](#footcite-ho2020denoising)</sup> and [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.html.md#deepinv.sampling.AncestralSolver).

* **Parameters:**
  * **timesteps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) *,* [*list*](https://docs.python.org/3.12/library/stdtypes.html#list)) – time steps at which the SDE will be discretized.
  * **t_start** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **t_end** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **num_steps** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **variance** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – the variance of the noise added on each step, either `"small"` (default) for the posterior variance $\tilde{\beta}_t$, or `"large"` for the variance $\beta_t$ of the forward transition.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – A random number generator for reproducibility.

<hr />

* **References:**

* <a id='footcite-ho2020denoising'>**[1]**</a> Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.
