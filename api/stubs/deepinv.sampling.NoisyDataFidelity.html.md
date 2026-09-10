# NoisyDataFidelity

### *class* deepinv.sampling.NoisyDataFidelity(d=None, weight=1.0, \*args, \*\*kwargs)

Bases: [`DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)

Preconditioned data fidelity term for noisy data $- \log p(y|x + \sigma(t) \omega)$
with $\omega\sim\mathcal{N}(0,\mathrm{I})$.

This is a base class for the conditional classes for approximating $\log p_t(y|x_t)$ used in diffusion
algorithms for inverse problems, in [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion).

It comes with a `.grad` method computing the score $\nabla_{x_t} \log p_t(y|x_t)$.

By default we have

$$
\nabla_{x_t} \log p(y|x + \sigma(t) \omega) = P(\forw{x_t'}-y),
$$

where $P$ is a preconditioner and $x_t'$ is an estimation of the image $x$.
By default, $P$ is defined as $A^\top$, $x_t' = x_t$ and this class matches the
[`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) class.

* **Parameters:**
  * **d** ([*deepinv.optim.Distance*](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance)) – Distance metric to use for the data fidelity term. Default to [`deepinv.optim.L2Distance`](https://deepinv.org/api/stubs/deepinv.optim.L2Distance.html.md#deepinv.optim.L2Distance).
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weighting factor for the data fidelity term. Default to 1.

#### diff(x, y, physics, \*args, \*\*kwargs)

Computes the difference $A(x) - y$ between the forward operator applied to the current iterate and the input data.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
* **Returns:**
  (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward(x, y, physics, \*args, \*\*kwargs)

Computes the data-fidelity term.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward operator
* **Returns:**
  (torch.Tensor) loss term.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, y, physics, \*args, \*\*kwargs)

Computes the gradient of the data-fidelity term.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model
* **Returns:**
  (torch.Tensor) data-fidelity term.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### precond(u, physics, \*args, \*\*kwargs)

The preconditioner $P$ for the data fidelity term. Default to $A^{\top}$.

* **Parameters:**
  * **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
* **Returns:**
  (torch.Tensor) preconditioned tensor $P(u)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-sampling-noisydatafidelity"></a>

## Examples using `NoisyDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
