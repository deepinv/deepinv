# HeunSolver

### *class* deepinv.sampling.HeunSolver(timesteps=None, t_start=None, t_end=None, num_steps=None, rng=None)

Bases: [`BaseSDESolver`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)

Heun solver for SDEs.

This solver uses the second-order Heun method to numerically integrate SDEs, defined as:

$$
\tilde{x}_{t+dt} &= x_t + f(x_t,t)dt + g(t) W_{dt} \\
x_{t+dt} &= x_t + \frac{1}{2}[f(x_t,t) + f(\tilde{x}_{t+dt},t+dt)]dt + \frac{1}{2}[g(t) + g(t+dt)] W_{dt}

$$

where $W_t$ is a Gaussian random variable with mean 0 and variance dt.

* **Parameters:**
  * **timesteps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*numpy.ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – The time steps at which to evaluate the solution.
  * **timesteps** – time steps at which the SDE will be discretized.
  * **t_start** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **t_end** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **num_steps** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – A random number generator for reproducibility.

#### NOTE
You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.
