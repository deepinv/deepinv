# DistributedStackedPhysics

### *class* deepinv.distributed.framework.DistributedStackedPhysics(ctx, num_operators, factory, , factory_kwargs=None, dtype=None, gather_strategy='concatenated', \*\*kwargs)

Bases: [`Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

This class distributes a *collection* of physics operators across multiple processes,
where each process owns a subset of the operators.

#### NOTE
It is intended to parallelize models naturally expressed as a stack/list of operators
(e.g., [`deepinv.physics.StackedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics) or an explicit Python list of
[`deepinv.physics.Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) objects) and is **not** meant to split a
single monolithic physics operator across ranks.

If your forward model is a single operator that can be decomposed into multiple
sub-operators, you can build any custom decomposition (e.g., build a
[`deepinv.physics.StackedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics)) and then pass that collection to
[`DistributedStackedPhysics`](#deepinv.distributed.framework.DistributedStackedPhysics) via the `factory` argument.

* **Parameters:**
  * **ctx** ([*DistributedContext*](https://deepinv.org/api/stubs/deepinv.distributed.DistributedContext.html.md#deepinv.distributed.DistributedContext)) – distributed context manager.
  * **num_operators** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – total number of physics operators.
  * **factory** (*Callable*) – factory function that creates physics operators. Should have signature `factory(index, device, factory_kwargs) -> Physics`.
  * **factory_kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *|* *None*) – shared data dictionary passed to factory function. Default is `None`.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) *|* *None*) – data type for operations. Default is `None`.
  * **gather_strategy** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – 

    strategy for gathering distributed results. Options are:
    - `'naive'`: Simple object serialization (best for small tensors)
    - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
    - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)

    Default is `'concatenated'`.

#### A(x, gather=True, reduce_op=None, force_input_grad_sync=False, return_graph_anchor=False, \*\*kwargs)

Apply forward operator to all distributed physics operators with automatic gathering.

Applies the forward operator $A(x)$ by computing local measurements and gathering
results from all ranks using the configured gather strategy.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input signal.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather results across ranks. If `False`, returns local measurements. Default is `True`.
  * **reduce_op** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – reduction operation to apply across ranks. Default is `None`.
  * **force_input_grad_sync** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – force synchronization of input gradients even
    in pure local mode (`gather=False` and `reduce_op=None`). Default is `False`.
  * **return_graph_anchor** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, also return the tensor used as graph
    anchor inside this call. Intended for advanced internal usage. Default is `False`.
  * **kwargs** – optional parameters for the forward operator.
* **Returns:**
  complete list of measurements from all operators (or local list if `reduce=False`).
* **Return type:**
  [*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) | [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) | [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### forward(x, gather=True, reduce_op=None, force_input_grad_sync=False, return_graph_anchor=False, \*\*kwargs)

Apply full forward model with sensor and noise models to the input signal and gather results.

$$
y = N(A(x))
$$

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input signal.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather results across ranks. If `False`, returns local measurements. Default is `True`.
  * **reduce_op** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – reduction operation to apply across ranks. Default is `None`.
  * **force_input_grad_sync** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – force synchronization of input gradients even
    in pure local mode (`gather=False` and `reduce_op=None`). Default is `False`.
  * **return_graph_anchor** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, also return the tensor used as graph
    anchor inside this call. Intended for advanced internal usage. Default is `False`.
  * **kwargs** – optional parameters for the forward model.
* **Returns:**
  complete list of noisy measurements from all operators.
