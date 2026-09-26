# BSREMIteration

### *class* deepinv.optim.optim_iterators.BSREMIteration(eps=1e-6, sensitivity_threshold=1e-2, cost_fn=None, \*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.md#deepinv.optim.OptimIterator)

Performs a single BSREM epoch, updating the estimate once per measurement subset.
See [`deepinv.optim.BSREM`](https://deepinv.org/api/stubs/deepinv.optim.BSREM.md#deepinv.optim.BSREM) for algorithm details.

* **Parameters:**
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Lower bound for division denominators and the reconstructed image. Default: `1e-6`.
  * **sensitivity_threshold** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Sensitivity threshold defining the reconstruction support. Default: `1e-2`.
  * **cost_fn** (*Callable*) – Custom cost function evaluated after each epoch. Default: `None`.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, sensitivities, \*args, \*\*kwargs)

Perform one Block Sequential Regularized EM epoch.

* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current iterate and estimated cost.
  * **cur_data_fidelity** ([*deepinv.optim.StackedPhysicsDataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.StackedPhysicsDataFidelity.md#deepinv.optim.StackedPhysicsDataFidelity)) – Data-fidelity terms corresponding to the physics subsets.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.md#deepinv.optim.Prior)) – Differentiable prior used for each subset update.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Algorithm parameters `"stepsize"`, `"lambda"`, and `"g_param"`.
  * **y** ([*deepinv.utils.TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.md#deepinv.utils.TensorList)) – Measurement subsets.
  * **physics** ([*deepinv.physics.StackedLinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.md#deepinv.physics.StackedLinearPhysics)) – Physics operators corresponding to the measurement subsets.
  * **sensitivities** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – Precomputed sensitivity maps $A_l^T\mathbf{1}$ for each subset.
* **Returns:**
  Dictionary `{"est": (x, None), "cost": F, "it": k + 1}` containing the updated iterate and estimated cost.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), None] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [int](https://docs.python.org/3.9/library/functions.html#int) | None]
