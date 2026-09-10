# DistributedStackedLinearPhysics

### *class* deepinv.distributed.framework.DistributedStackedLinearPhysics(ctx, num_operators, factory, , factory_kwargs=None, dtype=None, gather_strategy='concatenated', \*\*kwargs)

Bases: [`DistributedStackedPhysics`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedPhysics.html.md#deepinv.distributed.framework.DistributedStackedPhysics), [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Distributed linear physics operators.

This class extends [`DistributedStackedPhysics`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedPhysics.html.md#deepinv.distributed.framework.DistributedStackedPhysics) for linear operators. It provides distributed
operations that automatically handle communication and reductions.

#### NOTE
This class is intended to distribute a *collection* of linear operators (e.g.,
[`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics) or an explicit Python list of
[`deepinv.physics.LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics) objects) across ranks. It is **not** a
mechanism to shard a single linear operator internally.

If you have one linear physics operator that can naturally be split into multiple
operators, you must do that split yourself (build a stacked/list representation) and
provide those operators through the `factory`.

All linear operations (`A_adjoint`, `A_vjp`, etc.) support a `reduce_op` parameter:

- If `reduce_op='sum'` (default): The method computes the global result by performing a single all-reduce across all ranks.
- If `reduce_op=None`: The method computes only the local contribution from operators owned by this rank, without any inter-rank communication. This is useful for deferring reductions in custom algorithms. Beware, using `reduce_op=None` may lead to errors or incorrect results if the full global operation is required (e.g., for correct gradients in training) and should be used with caution.

* **Parameters:**
  * **ctx** ([*DistributedContext*](https://deepinv.org/api/stubs/deepinv.distributed.DistributedContext.html.md#deepinv.distributed.DistributedContext)) – distributed context manager.
  * **num_operators** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – total number of physics operators to distribute.
  * **factory** (*Callable*) – factory function that creates linear physics operators.
    Should have signature `factory(index: int, device: torch.device, factory_kwargs: dict | None) -> LinearPhysics`.
  * **factory_kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *|* *None*) – shared data dictionary passed to factory function for all operators. Default is `None`.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) *|* *None*) – data type for operations. Default is `None`.
  * **gather_strategy** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – strategy for gathering distributed results in forward operations.
    Options are `'naive'`, `'concatenated'`, or `'broadcast'`. Default is `'concatenated'`.

#### A_A_adjoint(y, gather=True, \*\*kwargs)

Compute global $A A^T$ operation with automatic reduction.

For stacked operators, this computes $A A^T y$ where $A^T y = \sum_i A_i^T y_i$
and then applies the forward operator to get $[A_1(A^T y), A_2(A^T y), \ldots, A_n(A^T y)]$.

#### NOTE
Unlike other operations, the adjoint step `A^T y` is always computed globally (with full
reduction across ranks) even when `gather=False`. This is because computing the correct
`A_A_adjoint` requires the full adjoint `sum_i A_i^T y_i`. The `gather` parameter only
controls whether the final forward operation `A(...)` is gathered across ranks.

* **Parameters:**
  * **y** ([*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) *|* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – full list of measurements from all operators.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather final results across ranks. If `False`, returns only local
    operators’ contributions (but still uses the global adjoint). Default is `True`.
  * **kwargs** – optional parameters for the operation.
* **Returns:**
  TensorList with entries $A_i A^T y$ for all operators (or local list if `gather=False`).
* **Return type:**
  [*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) | [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### A_adjoint(y, gather=True, reduce_op='sum', \*\*kwargs)

Compute global adjoint operation with automatic reduction.

Extracts local measurements, computes local adjoint contributions, and sum reduces
across all ranks to obtain the complete $A^T y = \sum_{i=1}^n A_i^T y_i$ where $A$ is the
stacked operator $A = [A_1, A_2, \ldots, A_n]$, $A_i$ and $y_i$ are the individual linear operators and measurements respectively.

* **Parameters:**
  * **y** ([*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) *|* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – full list of measurements from all operators.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather results across ranks. If False, returns local contribution. Default is `True`.
  * **reduce_op** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – reduction operation to apply across ranks. If `None`, no reduction (neither local or global) is performed, reduction must be performed manually to produce complete adjoint result $A^T y$. Default is `"sum"`.
  * **kwargs** – optional parameters for the adjoint operation.
* **Returns:**
  complete adjoint result $A^T y$ (or local contribution if gather=False).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint_A(x, gather=True, reduce_op='sum', \*\*kwargs)

Compute global $A^T A$ operation with automatic reduction.

Computes the complete normal operator $A^T A x = \sum_i A_i^T A_i x$ by
combining local contributions from all ranks.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather results across ranks. If False, returns local contribution. Default is `True`.
  * **reduce_op** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – reduction operation to apply across ranks. If `None`, no reduction (neither local or global) is performed, reduction must be performed manually to produce complete  result $A^T A x$. Default is `"sum"`.
  * **kwargs** – optional parameters for the operation.
* **Returns:**
  complete $A^T A x$ result (or local contribution if gather=False).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_dagger(y, solver='CG', max_iter=None, tol=None, verbose=False, , local_only=True, gather=True, \*\*kwargs)

Distributed pseudoinverse computation. This method provides two strategies:

1. **Local approximation** (`local_only=True`, default): Each rank computes the pseudoinverse
of its local operators independently, then averages the results with a single reduction.
This is efficient (minimal communication) but **provides only an approximation**.
In other words, for stacked operators this computes

$$
A^\dagger y = \frac{1}{n} \sum_i A_i^\dagger y_i
$$

2. **Global computation** (`local_only=False`): Uses the full least squares solver
with distributed [`A_adjoint_A()`](#deepinv.distributed.framework.DistributedStackedLinearPhysics.A_adjoint_A) and [`A_A_adjoint()`](#deepinv.distributed.framework.DistributedStackedLinearPhysics.A_A_adjoint) operations.
This computes the exact pseudoinverse but requires communication at every iteration.

* **Parameters:**
  * **y** ([*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) *|* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – measurements to invert.
  * **solver** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – least squares solver to use (only for `local_only=False`).
    Choose between `'CG'`, `'lsqr'`, `'BiCGStab'` and `'minres'`. Default is `'CG'`.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* *None*) – maximum number of iterations for least squares solver. Default is `None`.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* *None*) – relative tolerance for least squares solver. Default is `None`.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – print information (only on rank 0). Default is `False`.
  * **local_only** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True` (default), compute local daggers and sum-reduce (efficient).
    If `False`, compute exact global pseudoinverse with full communication (expensive). Default is `True`.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather results across ranks (only applies if local_only=True). Default is `True`.
  * **kwargs** – optional parameters for the forward operator.
* **Returns:**
  pseudoinverse solution. If `local_only=True`, returns approximation.
  If `local_only=False`, returns exact least squares solution.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_vjp(x, v, gather=True, reduce_op='sum', \*\*kwargs)

Compute global vector-Jacobian product with automatic reduction.

Extracts local cotangent vectors, computes local VJP contributions, and reduces
across all ranks to obtain the complete VJP.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor.
  * **v** ([*TensorList*](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) *|* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – full list of cotangent vectors from all operators.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather results across ranks. If False, returns local contribution. Default is `True`.
  * **reduce_op** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – reduction operation to apply across ranks. If `None`, no reduction (neither local or global) is performed, reduction must be performed manually to produce complete VJP result. Default is `"sum"`.
  * **kwargs** – optional parameters for the VJP operation.
* **Returns:**
  complete VJP result (or local contribution if gather=False).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### compute_sqnorm(x0, , max_iter=50, tol=1e-3, verbose=True, local_only=True, gather=True, \*\*kwargs)

Computes the squared spectral $\ell_2$ norm of the distributed operator.

This method provides two strategies:

1. **Local approximation** (`local_only=True`, default): Each rank computes the norm
of its local operators independently, then a single max-reduction provides an upper bound.
This is efficient (minimal communication) and valid for conservative estimates.
For stacked operators $A = [A_1; A_2; \ldots; A_n]$, we have
$\|A\|^2 \leq \sum_i \|A_i\|^2$, and we use $\max_i \|A_i\|^2$ as
a conservative upper bound.

2. **Global computation** (`local_only=False`): Uses the full distributed [`A_adjoint_A()`](#deepinv.distributed.framework.DistributedStackedLinearPhysics.A_adjoint_A)
with communication at every power iteration. This computes the exact norm but is
communication-intensive.

* **Parameters:**
  * **x0** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an unbatched tensor sharing its shape, dtype and device with the initial iterate.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations for power method. Default is `50`.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – relative variation criterion for convergence. Default is `1e-3`.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – print information (only on rank 0). Default is `True`.
  * **local_only** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True` (default), compute local norms and max-reduce (efficient).
    If `False`, compute exact global norm with full communication (expensive). Default is `True`.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather results across ranks (only applies if local_only=True). Default is `True`.
  * **kwargs** – optional parameters for the forward operator.
* **Returns:**
  Squared spectral norm. If `local_only=True`, returns upper bound.
  If `local_only=False`, returns exact value.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-distributed-framework-distributedstackedlinearphysics"></a>

## Examples using `DistributedStackedLinearPhysics`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Physics Operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
