# BregmanL2

### *class* deepinv.optim.BregmanL2

Bases: [`Bregman`](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)

Module for the L2 norm as Bregman potential $\phi(x) = \frac{1}{2} \|x\|_2^2$.
The corresponding Bregman divergence is the squared Euclidean distance $D(x,y) = \frac{1}{2} \|x-y\|_2^2$.

#### conjugate(x)

Computes the convex conjugate potential $\phi^*(y) = \frac{1}{2} \|y\|_2^2$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the conjugate is computed.
* **Returns:**
  (torch.Tensor) conjugate potential $\phi^*(y)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### div(x, y, \*args, \*\*kwargs)

Computes the Bregman divergence with potential $\phi$. Here falls back to the L2 distance.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the divergence is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $y$ at which the divergence is computed.
* **Returns:**
  (torch.Tensor) divergence $\phi(x) - \phi(y) - \langle \nabla \phi(y), x-y$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### fn(x)

Computes the L2 norm potential $\phi(x) = \frac{1}{2} \|x\|_2^2$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the potential is computed.
* **Returns:**
  (torch.Tensor) potential $h(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, \*args, \*\*kwargs)

Calculates the gradient of the L2 norm $\nabla \phi(x) = x$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
* **Returns:**
  (torch.Tensor) gradient $\nabla_x \phi$, computed in $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad_conj(x, \*args, \*\*kwargs)

Calculates the gradient of the conjugate of the L2 norm $\nabla \phi^*(x) = x$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
* **Returns:**
  (torch.Tensor) gradient $\nabla_x \phi^*$, computed in $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
