# FISTAIteration

### *class* deepinv.optim.optim_iterators.FISTAIteration(a=3, \*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for fast iterative soft-thresholding (FISTA).

Class for a single iteration of the FISTA algorithm for minimizing $f(x) + \lambda \regname(x)$ as proposed by Chambolle and Dossal<sup>[1](#footcite-chambolle2015convergence)</sup>.

The iteration is given by

$$
u_{k} &= z_k -  \gamma \nabla f(z_k) \\
x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(u_k) \\
z_{k+1} &= x_{k+1} + \alpha_k (x_{k+1} - x_k)

$$

where $\gamma$ is a stepsize that should satisfy $\gamma \leq 1/\operatorname{Lip}(\|\nabla f\|)$ and
$\alpha_k = (k+a-1)/(k+a)$, with $a$ a parameter that should be strictly greater than 2.

<hr />

* **References:**

* <a id='footcite-chambolle2015convergence'>**[1]**</a> Antonin Chambolle and Charles H Dossal. On the convergence of the iterates of” fista”. *Journal of Optimization Theory and Applications*, 166(3):25, 2015.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, \*args, \*\*kwargs)

Forward pass of an iterate of the FISTA algorithm.

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
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [int](https://docs.python.org/3.9/library/functions.html#int)]
