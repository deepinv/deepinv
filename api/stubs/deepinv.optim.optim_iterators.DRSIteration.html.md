# DRSIteration

### *class* deepinv.optim.optim_iterators.DRSIteration(\*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for Douglas-Rachford Splitting.

Class for a single iteration of the Douglas-Rachford Splitting (DRS) algorithm for minimizing
$f(x) + \lambda \regname(x)$.

If the attribute `g_first` is set to False (by default), the iteration is given by

$$
u_{k+1} &= \operatorname{prox}_{\gamma f}(z_k) \\
x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(2*u_{k+1}-z_k) \\
z_{k+1} &= z_k + \beta (x_{k+1} - u_{k+1})

$$

where $\gamma>0$ is a stepsize and $\beta>0$ is a relaxation parameter.

If the attribute `g_first` is set to True, the functions $f$ and $\regname$ are inverted in the previous iteration.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, \*args, \*\*kwargs)

Single iteration of the DRS algorithm.

* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current iterate and the estimated cost.
  * **cur_data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Instance of the DataFidelity class defining the current data_fidelity.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Instance of the Prior class defining the current prior.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current parameters of the algorithm.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Instance of the physics modeling the observation.
* **Returns:**
  Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
