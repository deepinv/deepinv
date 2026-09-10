# DistributedDataFidelity

### *class* deepinv.distributed.framework.DistributedDataFidelity(ctx, data_fidelity, num_operators=None, , factory_kwargs=None, reduction='sum')

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Distributed data fidelity term for use with distributed physics operators.

This class wraps a standard DataFidelity object and makes it compatible with
DistributedStackedLinearPhysics by implementing efficient distributed computation patterns.
It computes data fidelity terms and gradients using local operations followed by
a single reduction, avoiding redundant communication.

The key operations are:

- `fn(x, y, physics)`: Computes the data fidelity $\sum_i d(A_i(x), y_i)$
- `grad(x, y, physics)`: Computes the gradient $\sum_i A_i^T \nabla d(A_i(x), y_i)$

Both operations use an efficient pattern:

1. Compute local forward operations $A_i$
2. Apply distance function and compute gradients locally
3. Perform a single reduction across ranks

* **Parameters:**
  * **ctx** ([*DistributedContext*](https://deepinv.org/api/stubs/deepinv.distributed.DistributedContext.html.md#deepinv.distributed.DistributedContext)) – distributed context manager.
  * **data_fidelity** ([*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *|* *Callable*) – either a DataFidelity instance
    or a factory function that creates DataFidelity instances for each operator.
    The factory should have signature
    `factory(index: int, device: torch.device, factory_kwargs: dict | None) -> DataFidelity`.
  * **num_operators** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* *None*) – number of operators (required if data_fidelity is a factory). Default is `None`.
  * **factory_kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *|* *None*) – shared data dictionary passed to factory function for all operators. Default is `None`.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – reduction mode matching the distributed physics. Options are `'sum'` or `'mean'`.
    Default is `'sum'`.

#### fn(x, y, physics, gather=True, \*args, \*\*kwargs)

Compute the distributed data fidelity term.

For distributed physics with operators $\{A_i\}$ and measurements $\{y_i\}$,
computes:

$$
f(x) = \sum_i d(A_i(x), y_i)
$$

This is computed efficiently by:

1. Each rank computes $A_i(x)$ for its local operators
2. Each rank computes $\sum_{i \in \text{local}} d(A_i(x), y_i)$
3. Results are reduced across all ranks

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input signal at which to evaluate the data fidelity.
  * **y** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – measurements (TensorList or list of tensors).
  * **physics** ([*DistributedStackedLinearPhysics*](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedLinearPhysics.html.md#deepinv.distributed.framework.DistributedStackedLinearPhysics)) – distributed physics operator.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather (reduce) results across ranks. Default is `True`.
  * **args** – additional positional arguments passed to the distance function.
  * **kwargs** – additional keyword arguments passed to the distance function.
* **Returns:**
  scalar data fidelity value.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, y, physics, gather=True, \*args, \*\*kwargs)

Compute the gradient of the distributed data fidelity term.

For distributed physics with operators $\{A_i\}$ and measurements $\{y_i\}$,
computes:

$$
\nabla_x f(x) = \sum_i \frac{\partial A_i}{\partial \x} \nabla d(A_i(x), y_i)
$$

This is computed efficiently by:

1. Each rank computes $A_i(x)$ for its local operators
2. Each rank computes $\nabla d(A_i(x), y_i)$ for its local operators
3. Each rank computes $\sum_{i \in \text{local}} \frac{\partial A_i}{\partial \x} \nabla d(A_i(x), y_i)$ using A_vjp_local
4. Results are reduced across all ranks

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input signal at which to compute the gradient.
  * **y** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – measurements (TensorList or list of tensors).
  * **physics** ([*DistributedStackedLinearPhysics*](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedLinearPhysics.html.md#deepinv.distributed.framework.DistributedStackedLinearPhysics)) – distributed physics operator.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather (reduce) results across ranks. Default is `True`.
  * **args** – additional positional arguments passed to the distance function gradient.
  * **kwargs** – additional keyword arguments passed to the distance function gradient.
* **Returns:**
  gradient with same shape as x.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, y, physics, \*args, gamma=1.0, \*\*kwargs)

Compute proximal step for distributed data-fidelity.

Currently supported when a single shared DataFidelity object is used for all
operators (the standard unfolded setup). In that case, the prox is delegated
to the wrapped DataFidelity with the distributed physics object.

<a id="sphx-glr-backref-deepinv-distributed-framework-distributeddatafidelity"></a>

## Examples using `DistributedDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div>
<!-- thumbnail-parent-div-close --></div>
