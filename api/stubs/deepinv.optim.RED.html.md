# RED

### *class* deepinv.optim.RED(denoiser, \*args, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

Regularization-by-Denoising (RED) prior $\nabla \reg{x} = x - \operatorname{D}_{\sigma}(x)$.

* **Parameters:**
  **denoiser** (*Callable*) – Denoiser $\operatorname{D}_{\sigma}$.

#### grad(x, sigma_denoiser, \*args, \*\*kwargs)

Calculates the gradient of the prior term $\regname$ at $x$.
By default, the gradient is computed using automatic differentiation.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) gradient $\nabla_x g$, computed in $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-red"></a>

## Examples using `RED`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of :footciteromano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) :footcitehurault2021gradient which provides an explicit prior.">  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
