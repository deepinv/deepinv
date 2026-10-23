# NegEntropy

### *class* deepinv.optim.NegEntropy

Bases: [`Bregman`](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)

Module for the using negative entropy as Bregman potential $\phi(x) = \sum_i x_i \log x_i$.

The corresponding Bregman divergence is the Kullback-Leibler divergence $D(x,y) = \sum_i x_i \log(x_i / y_i) - x_i + y_i$.

#### conjugate(x)

Computes the convex conjugate potential $\phi^*(y) = \sum_i y_i \log y_i$.
The input $x$ must be negative.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the conjugate is computed.
* **Returns:**
  (torch.Tensor) conjugate potential $\phi^*(y)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### fn(x)

Computes negative entropy potential $\phi(x) = \sum_i x_i \log x_i$.
The input $x$ must be postive.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the potential is computed.
* **Returns:**
  (torch.Tensor) potential $\phi(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, \*args, \*\*kwargs)

Calculates the gradient of negative entropy $\nabla \phi(x) = 1 + \log x$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
* **Returns:**
  (torch.Tensor) gradient $\nabla_x \phi$, computed in $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad_conj(x, \*args, \*\*kwargs)

Calculates the gradient of the conjugate of negative entropy $\nabla \phi^*(x) = 1 + \log x$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
* **Returns:**
  (torch.Tensor) gradient $\nabla_x \phi^*$, computed in $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
