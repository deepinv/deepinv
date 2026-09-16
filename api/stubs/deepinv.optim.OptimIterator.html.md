# OptimIterator

### *class* deepinv.optim.OptimIterator(g_first=False, cost_fn=None, has_cost=True, \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class for optimization iterators.

An optim iterator is an object that implements a fixed point iteration for minimizing the sum of two functions
$F = f + \lambda \regname$ where $f$ is a data-fidelity term  that will be modeled by an instance of physics
and $\regname$ is a regularizer. The fixed point iteration takes the form

$$
\qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, \regname, A, y, ...)

$$

where $x$ is a “primal” variable converging to the solution of the minimization problem, and
$z$ is a “dual” variable.

#### NOTE
By an abuse of terminology, we call “primal” and “dual” variables the variables that are updated
at each step and which may correspond to the actual primal and dual variables from
(for instance in the case of the PD algorithm), but not necessarily (for instance in the case of the
PGD algorithm).

The implementation of the fixed point algorithm in [`deepinv.optim.FixedPoint`](https://deepinv.org/api/stubs/deepinv.optim.FixedPoint.html.md#deepinv.optim.FixedPoint) is split in two steps, alternating between
a step on $f$ and a step on $\regname$, that is for $k=1,2,...$

$$
z_{k+1} = \operatorname{step}_f(x_k, z_k, y, A, ...)\\
x_{k+1} = \operatorname{step}_{\regname}(x_k, z_k, y, A, ...)

$$

where $\operatorname{step}_f$ and $\operatorname{step}_{\regname}$ are the steps on f and g respectively.

* **Parameters:**
  * **g_first** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, the algorithm starts with a step on g and finishes with a step on f.
  * **cost_fn** (*Callable*) – function that returns the function F to be minimized at each iteration. Default: None.
  * **has_cost** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, the cost function $D$ is computed at each iteration. Default: True.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, \*args, \*\*kwargs)

General form of a single iteration of splitting algorithms for minimizing $F =  f + \lambda \regname$, alternating
between a step on $f$ and a step on $\regname$.
The primal and dual variables as well as the estimated cost at the current iterate are stored in a dictionary
`X` of the form `{'est': (x,z), 'cost': F}`.

* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *]*) – Dictionary containing the current iterate and the estimated cost.
  * **cur_data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – Instance of the DataFidelity class defining the current data_fidelity.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)) – Instance of the Prior class defining the current prior.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the current parameters of the algorithm.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Instance of the physics modeling the observation.
* **Returns:**
  Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### relaxation_step(u, v, beta)

Performs a relaxation step of the form $\beta u + (1-\beta) v$.

* **Parameters:**
  * **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – First tensor.
  * **v** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Second tensor.
  * **beta** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Relaxation parameter.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) Relaxed tensor.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-optimiterator"></a>

## Examples using `OptimIterator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
