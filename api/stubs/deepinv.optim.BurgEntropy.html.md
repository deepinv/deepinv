# BurgEntropy

### *class* deepinv.optim.BurgEntropy

Bases: [`Bregman`](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)

Module for the using Burg’s entropy as Bregman potential $\phi(x) = - \sum_i \log x_i$.

The corresponding Bregman divergence is the Itakura-Saito distance $D(x,y) = \sum_i x_i / y_i - \log(x_i / y_i) - 1$.
As shown in Bolte *et al.*<sup>[1](#footcite-bolte2016descent)</sup>, it is the Bregman potential to use for performing mirror descent on the Poisson likelihood [`deepinv.optim.data_fidelity.PoissonLikelihood`](https://deepinv.org/api/stubs/deepinv.optim.PoissonLikelihood.html.md#deepinv.optim.PoissonLikelihood).

<hr />

* **References:**

* <a id='footcite-bolte2016descent'>**[1]**</a> Jérôme Bolte, Heinz Bauschke, and Marc Teboulle. A descent lemma beyond lipschitz gradient continuity: first-order methods revisited and applications. *Mathematics of Operations Research*, 42:, 07 2016. [doi:10.1287/moor.2016.0817](https://doi.org/10.1287/moor.2016.0817).

#### conjugate(x)

Computes the convex conjugate potential $\phi^*(y) = - - \sum_i \log (-x_i)$.
The input $x$ must be negative.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the conjugate is computed.
* **Returns:**
  (torch.Tensor) conjugate potential $\phi^*(y)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### fn(x)

Computes Burg’s entropy potential $\phi(x) = - \sum_i \log x_i$.
The input $x$ must be postive.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the potential is computed.
* **Returns:**
  (torch.Tensor) potential $h(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, \*args, \*\*kwargs)

Calculates the gradient of Burg’s entropy $\nabla \phi(x) = - 1 / x$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
* **Returns:**
  (torch.Tensor) gradient $\nabla_x \phi$, computed in $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad_conj(x, \*args, \*\*kwargs)

Calculates the gradient of the conjugate of Burg’s entropy $\nabla h^*(x) = - 1 / x$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
* **Returns:**
  (torch.Tensor) gradient $\nabla_x h^*$, computed in $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-burgentropy"></a>

## Examples using `BurgEntropy`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
