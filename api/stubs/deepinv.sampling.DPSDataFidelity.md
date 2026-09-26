# DPSDataFidelity

### *class* deepinv.sampling.DPSDataFidelity(denoiser=None, weight=1.0, clip=None, guidance='norm', \*args, \*\*kwargs)

Bases: [`NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.md#deepinv.sampling.NoisyDataFidelity)

Diffusion posterior sampling data-fidelity term.

This corresponds to the $p(y|x_t)$ approximation proposed in [[29](https://deepinv.org/user_guide/other/biblio.md#id65)].
For the VE parametrization $x_t=x_0+\sigma_t\omega$, DPS replaces
$p(x_0|x_t)$ by a Dirac mass at the denoised posterior mean:

$$
p(x_0|x_t)
\approx \delta\!\left(x_0-D(x_t,\sigma_t)\right).
$$

Two guidance strengths are available, selected with `guidance`.
`guidance="norm"` (the default) follows [[29](https://deepinv.org/user_guide/other/biblio.md#id65)] and normalizes
the residual by its own norm,

$$
-\nabla_x \log p_t(y|x) \approx \lambda \nabla_x \| \forw{\denoiser{x}{\sigma}} - y \|,
$$

which is the step size $\zeta/\|y - A D(x_t,\sigma_t)\|$ of the original paper.
`guidance="annealed"` instead uses the Gaussian negative log-likelihood with the
annealed variance $\sigma_y^2+\sigma_t^2$,

$$
-\nabla_x \log p_t(y|x) \approx \lambda \nabla_x
\frac{\| \forw{\denoiser{x}{\sigma}} - y \|^2}{2\left(\sigma_y^2+\sigma_t^2\right)},
$$

where $\sigma = \sigma(t)$ is the noise level and $\lambda$
controls the strength of the approximation.

#### NOTE
The two options put `weight` on very different scales. `"norm"` carries no
noise variance, so $\lambda$ has to absorb a factor of order
$\|y - A D(x_t,\sigma_t)\|/\sigma_y^2$, which is typically in the hundreds.
`"annealed"` shares the guidance strength of
[`deepinv.sampling.ALDDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.ALDDataFidelity.md#deepinv.sampling.ALDDataFidelity), so $\lambda\approx 1$ is the
natural choice, consistent with the other noisy data-fidelity terms.

#### SEE ALSO
This class can be used for building custom DPS-based diffusion models.
A self-contained implementation of the original DPS algorithm can be
found in [`deepinv.sampling.DPS`](https://deepinv.org/api/stubs/deepinv.sampling.DPS.md#deepinv.sampling.DPS).

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.md#deepinv.models.Denoiser)) – Denoiser network
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weighting factor for the data fidelity term. Default to 1.0 .
  * **clip** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
  * **guidance** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Either `"norm"` (default), the residual norm of
    [[29](https://deepinv.org/user_guide/other/biblio.md#id65)], or `"annealed"`, the Gaussian negative
    log-likelihood with variance $\sigma_y^2+\sigma_t^2$, for which
    `weight` is on the same scale as the other noisy data-fidelity terms.

#### forward(x, y, physics, sigma, \*args, get_model_outputs=False, \*\*kwargs)

Returns the loss term
$\lambda \| \forw{\denoiser{x}{\sigma}} - y \|$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – forward operator
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – standard deviation of the noise.
  * **get_model_outputs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, also return the denoised output along with the loss. Default to `False`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) or tuple of [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss term (and denoised output if `get_model_outputs` is `True`).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### grad(x, y, physics, sigma, \*args, get_model_outputs=False, \*\*kwargs)

Computes the gradient $-\nabla_{x_t} \log p_t(y|x_t) \approx \lambda \nabla_{x_t} \| \forw{\denoiser{x}{\sigma}} - y \|$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input data.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – physics model
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Standard deviation of the noise.
  * **get_model_outputs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, also return the denoised output along with the score. Default to `False`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) or tuple of [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) score term (and denoised output if `get_model_outputs` is `True`).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

<a id="sphx-glr-backref-deepinv-sampling-dpsdatafidelity"></a>

## Examples using `DPSDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusers_thumb.png)

[Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse](https://deepinv.org/auto_examples/sampling/demo_diffusers.md)

  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_sde_thumb.png)

[Building your diffusion posterior sampling method using SDEs](https://deepinv.org/auto_examples/sampling/demo_diffusion_sde.md)

  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in chung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_dps_thumb.png)

[DPS – Posterior Sampling with Diffusion Models](https://deepinv.org/auto_examples/sampling/demo_dps.md)

  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">![](auto_examples/sampling/images/thumb/sphx_glr_demo_flow_matching_thumb.png)

[Flow-Matching for posterior sampling and unconditional generation](https://deepinv.org/auto_examples/sampling/demo_flow_matching.md)

  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
