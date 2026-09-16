# Distance

### *class* deepinv.optim.Distance(d=None)

Bases: [`Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential)

Distance $\distance{x}{y}$.

This is the base class for a distance $\distance{x}{y}$ between a variable $x$ and an observation $y$.
Comes with methods to compute the distance gradient, proximal operator or convex conjugate with respect to the variable $x$.

#### WARNING
All variables have a batch dimension as first dimension.

* **Parameters:**
  **d** (*Callable*) – distance function $\distance{x}{y}$. Outputs a tensor of size `B`, the size of the batch. Default: None.

#### fn(x, y, \*args, \*\*kwargs)

Computes the distance $\distance{x}{y}$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observation $y$.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) distance $\distance{x}{y}$ of size `B` with `B` the size of the batch.
* **Raises:**
  [**NotImplementedError**](https://docs.python.org/3.9/library/exceptions.html#NotImplementedError) – if the distance was instantiated without a distance function `d`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward(x, y, \*args, \*\*kwargs)

Computes the value of the distance $\distance{x}{y}$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Observation $y$.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) distance $\distance{x}{y}$ of size `B` with `B` the size of the batch.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
