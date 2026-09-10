# DPSDataFidelity

### *class* deepinv.sampling.DPSDataFidelity(denoiser=None, weight=1.0, clip=None, \*args, \*\*kwargs)

Bases: [`NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.html.md#deepinv.sampling.NoisyDataFidelity)

Diffusion posterior sampling data-fidelity term.

This corresponds to the $p(y|x_t)$ approximation proposed in [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687).

$$
\nabla_x \log p_t(y|x) = \nabla_x \frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|

$$

where $\sigma = \sigma(t)$ is the noise level, $m$ is the number of measurements (size of $y$),
and $\lambda$ controls the strength of the approximation.

#### SEE ALSO
This class can be used for building custom DPS-based diffusion models.
A self-contained implementation of the original DPS algorithm can be find in [`deepinv.sampling.DPS`](https://deepinv.org/api/stubs/deepinv.sampling.DPS.html.md#deepinv.sampling.DPS).

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – Denoiser network
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weighting factor for the data fidelity term. Default to 1.0 .
  * **clip** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.

#### forward(x, y, physics, sigma, \*args, get_model_outputs=False, \*\*kwargs)

Returns the loss term $\frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward operator
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – standard deviation of the noise.
  * **get_model_outputs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, also return the denoised output along with the loss. Default to `False`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) or tuple of [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss term (and denoised output if `get_model_outputs` is `True`).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### grad(x, y, physics, sigma, \*args, get_model_outputs=False, \*\*kwargs)

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Standard deviation of the noise.
  * **get_model_outputs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, also return the denoised output along with the score. Default to `False`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) or tuple of [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) score term (and denoised output if `get_model_outputs` is `True`).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

<a id="sphx-glr-backref-deepinv-sampling-dpsdatafidelity"></a>

## Examples using `DPSDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
