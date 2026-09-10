# OSEMIteration

### *class* deepinv.optim.optim_iterators.OSEMIteration(eps=1e-6, cost_fn=None, \*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)

Performs a single iteration of the OSEM algorithm, which is a classic baseline reconstruction method for inverse problems with Poisson noise statistics.
Note that [`deepinv.optim.optim_iterators.MLEMIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.MLEMIteration.html.md#deepinv.optim.optim_iterators.MLEMIteration) is a special case with one subset only.
More details on the algorithm can be found in the documentation of the
[`deepinv.optim.optimizers.OSEM`](https://deepinv.org/api/stubs/deepinv.optim.OSEM.html.md#deepinv.optim.OSEM) optimizer.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, sensitivities, \*args, \*\*kwargs)

Perform one Ordered-Subsets Expectation-Maximization step.

* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current iterate and the estimated cost.
  * **cur_data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Instance of the DataFidelity class defining the current data fidelity.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Instance of the Prior class defining the current prior.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current parameters of the algorithm.
  * **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList)) – Measurement subsets.
  * **physics** ([*deepinv.physics.StackedLinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics)) – Physics operators corresponding to the measurement subsets.
  * **sensitivities** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – Precomputed sensitivity maps $A_l^T \mathbf{1}$ for each subset.
* **Returns:**
  Dictionary `{"est": (x, None), "cost": F, "it": k + 1}` containing the updated iterate and estimated cost.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), None] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [int](https://docs.python.org/3.9/library/functions.html#int) | None]
