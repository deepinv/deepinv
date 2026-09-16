# Bregman

### *class* deepinv.optim.Bregman(phi=None)

Bases: [`Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential)

Module for the Bregman framework with convex Bregman potential $\phi$.
Comes with methods to compute the potential, its gradient, its conjugate, its gradient and its Bregman divergence.

* **Parameters:**
  **h** (*Callable*) – Potential function $\phi(x)$ to be used in the Bregman framework.

#### MD_step(x, grad, \*args, gamma=1.0, \*\*kwargs)

Performs a Mirror Descent step $x = \nabla \phi^*(\nabla \phi(x) - \gamma \nabla f(x))$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the step is performed.
  * **grad** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Gradient of the minimized function at $x$.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Step size.
* **Returns:**
  (torch.Tensor) updated variable $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### div(x, y, \*args, \*\*kwargs)

Computes the Bregman divergence $D_\phi(x,y)$ with Bregman potential $\phi$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Left variable $x$ at which the divergence is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Right variable $y$ at which the divergence is computed.
* **Returns:**
  (torch.Tensor) divergence $h(x) - h(y) - \langle \nabla h(y), x-y  \rangle$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-bregman"></a>

## Examples using `Bregman`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
