# DDIMSolver

### *class* deepinv.sampling.DDIMSolver(timesteps=None, t_start=None, t_end=None, num_steps=None, eta=0.0, variance='small', rng=None)

Bases: [`AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.md#deepinv.sampling.AncestralSolver)

DDIM solver for reverse-time diffusion SDEs.

Sampler of DDIM <sup>[1](#footcite-song2020denoising)</sup> with the stochasticity parameter $\eta$.
The default $\eta = 0$ gives the deterministic DDIM sampler and $\eta = 1$ gives the DDPM sampler,
see [`deepinv.sampling.DDPMSolver`](https://deepinv.org/api/stubs/deepinv.sampling.DDPMSolver.md#deepinv.sampling.DDPMSolver).

The `alpha` of the SDE is ignored, and replaced on each step of [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.md#deepinv.sampling.AncestralSolver) by

$$
\alpha_\eta = \frac{\log\left(1 - \eta^2 (1 - r^2)\right)}{2 \log r}, \quad r = \frac{\sigma(t+dt)}{\sigma(t)}.

$$

This relation depends on the step: $\alpha_\eta = \eta$ for $\eta \in \{0, 1\}$, and $\alpha_\eta$ tends to $\eta^2$ for small steps.

With `variance="large"`, the noise is scaled by $s(t) \sigma(t) / (s(t+dt) \sigma(t+dt))$, see [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.md#deepinv.sampling.AncestralSolver).
For $\eta = 1$, this is the DDPM sampler with the variance $\beta_t$ of the forward transition.

* **Parameters:**
  * **timesteps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) *,* [*list*](https://docs.python.org/3.12/library/stdtypes.html#list)) – time steps at which the SDE will be discretized.
  * **t_start** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **t_end** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **num_steps** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **eta** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – the stochasticity parameter $\eta \in [0, 1]$ of DDIM. Default to `0`.
  * **variance** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – the variance of the noise added on each step, either `"small"` (default) or `"large"`, see [`deepinv.sampling.AncestralSolver`](https://deepinv.org/api/stubs/deepinv.sampling.AncestralSolver.md#deepinv.sampling.AncestralSolver).
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – A random number generator for reproducibility.

<hr />

* **References:**

* <a id='footcite-song2020denoising'>**[1]**</a> Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *International Conference on Learning Representations*. 2020.
