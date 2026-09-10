# MDIteration

### *class* deepinv.optim.optim_iterators.MDIteration(bregman_potential=None, \*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for Mirror Descent.

Class for a single iteration of the mirror descent (GD) algorithm for minimising $f(x) + \lambda g(x)$.

For a given convex potential $h$, the iteration is given by

$$
v_{k} &= \nabla f(x_k) + \lambda \nabla g(x_k) \\
x_{k+1} &= \nabla h^*(\nabla h(x_k) - \gamma v_{k})

$$

where $\gamma$ is a stepsize.

* **Parameters:**
  **bregman_potential** ([*deepinv.optim.Bregman*](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)) – Instance of the Bregman class defining the Bregman potential used to compute the mirror descent step.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, \*args, \*\*kwargs)

Single mirror descent iteration on the objective $f(x) + \lambda g(x)$.
The Bregman potential, which is an intance of the deepinv.optim.Bregman class, is used to compute the mirror descent step.

* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current iterate $x_k$.
  * **cur_data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Instance of the DataFidelity class defining the current data_fidelity.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Instance of the Prior class defining the current prior.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current parameters of the algorithm.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Instance of the `Physics` class defining the current physics.
* **Returns:**
  Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
