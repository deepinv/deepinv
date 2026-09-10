# ADMMIteration

### *class* deepinv.optim.optim_iterators.ADMMIteration(\*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Iterator for alternating direction method of multipliers.

Class for a single iteration of the Alternating Direction Method of Multipliers (ADMM) algorithm for
minimising $f(x) + \lambda \regname(x)$.

If the attribute `g_first` is set to False (by default),
the iteration is (see Boyd *et al.*<sup>[1](#footcite-boyd2011distributed)</sup>):

$$
u_{k+1} &= \operatorname{prox}_{\gamma f}(x_k - z_k) \\
x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(u_{k+1} + z_k) \\
z_{k+1} &= z_k + \beta (u_{k+1} - x_{k+1})

$$

where $\gamma>0$ is a stepsize and $\beta>0$ is a relaxation parameter.

If the attribute `g_first` is set to `True`, the functions $f$ and $\regname$ are
inverted in the previous iteration.

<hr />

* **References:**

* <a id='footcite-boyd2011distributed'>**[1]**</a> Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, Jonathan Eckstein, and others. Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends® in Machine learning*, 3(1):1–122, 2011.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics)

Single iteration of the ADMM algorithm.

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
