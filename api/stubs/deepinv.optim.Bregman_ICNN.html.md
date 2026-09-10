# Bregman_ICNN

### *class* deepinv.optim.Bregman_ICNN(forw_model, conj_model=None)

Bases: [`Bregman`](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)

Module for the using a deep ICNN as Bregman potential.

#### conjugate(x)

Computes the convex conjugate potential.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the conjugate is computed.
* **Returns:**
  (torch.Tensor) conjugate potential $\phi^*(y)$.

#### fn(x)

Computes the Bregman potential.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the potential is computed.
* **Returns:**
  (torch.Tensor) potential $\phi(x)$.
