# SDEOutput

### *class* deepinv.sampling.SDEOutput(sample, trajectory, timesteps, nfe)

Bases: [`dict`](https://docs.python.org/3.9/library/stdtypes.html#dict)

A container for storing the output of an SDE solver, that behaves like a `dict` but allows access with the attribute syntax.

Attributes:
:attr torch.Tensor sample: the final samples of the sampling process, of shape `(B, C, H, W)`.
:attr torch.Tensor trajectory: the trajectory of the sampling process, of shape `(num_steps, B, C, H, W)` if `full_trajectory` is `True`, otherwise of shape `(B, C, H, W)`.
:attr torch.Tensor timesteps: the time steps at which the samples were taken, of shape `(num_steps,)`.
:attr int nfe: the number of function evaluations performed during the integration.
