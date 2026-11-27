# DiffPIRDataFidelity

### *class* deepinv.sampling.DiffPIRDataFidelity(denoiser=None, data_fidelity=None, sigma=0.05, weight=7.0, clip=None, eps=1e-6)

Bases: [`NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.md#deepinv.sampling.NoisyDataFidelity)

Diffusion PnP Image Restoration (DiffPIR) noisy data-fidelity term.

This corresponds to the proximal data-consistency step proposed in Zhu *et al.*<sup>[1](#footcite-zhu2023denoising)</sup>.

This class can be used for building custom DiffPIR-based diffusion models with
[`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.md#deepinv.sampling.PosteriorDiffusion). A self-contained implementation of
DiffPIR is available in [`deepinv.sampling.DiffPIR`](https://deepinv.org/api/stubs/deepinv.sampling.DiffPIR.md#deepinv.sampling.DiffPIR).

It approximates the gradient of the log-likelihood $\log p_t(y \vert x_t)$ as

$$
\nabla_{x_t} \log p_t(y \vert x_t) \approx
\frac{\hat{x}_{0, t} - \denoiser{x_t}{\sigma_t}}{\sigma_t^2},
$$

where

$$
\hat{x}_{0, t} = \operatorname{prox}_{\frac{1}{\rho_t} f}\left(\denoiser{x_t}{\sigma_t}\right),
\qquad \rho_t = \lambda \frac{\sigma_y^2}{\sigma_t^2}.
$$

See [`deepinv.sampling.DiffPIR`](https://deepinv.org/api/stubs/deepinv.sampling.DiffPIR.md#deepinv.sampling.DiffPIR) for more details on this method.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.md#deepinv.models.Denoiser)) – Denoiser network. It may be left as `None`
    when passed to [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.md#deepinv.sampling.PosteriorDiffusion), which supplies its denoiser.
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.md#deepinv.optim.DataFidelity)) – Measurement data-fidelity term $f(\cdot, y)$
    providing a `.prox` method. Default: [`deepinv.optim.L2`](https://deepinv.org/api/stubs/deepinv.optim.L2.md#deepinv.optim.L2).
  * **sigma** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Fallback measurement noise standard deviation $\sigma_y$ used when
    `physics.noise_model` does not define `sigma`. Default: `0.05`.
  * **weight** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Regularization hyperparameter $\lambda$ in
    $\rho_t = \lambda \sigma_y^2 / \sigma_t^2$. Default: `7.0`.
  * **clip** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – If not `None`, clip the denoised output into `[clip[0], clip[1]]`
    interval before the proximal step. Default: `None`.
  * **eps** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Numerical floor to avoid division by zero. Default: `1e-6`.

<hr />

* **References:**

* <a id='footcite-zhu2023denoising'>**[1]**</a> Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 1219–1229. 2023.

#### grad(x, y, physics, sigma, \*args, get_model_outputs=False, \*\*kwargs)

Compute the DiffPIR noisy data-fidelity gradient approximation.

$$
-\nabla_{x_t} \log p_t(y \vert x_t) \approx
\frac{1}{\sigma_t^2} \left(\denoiser{x_t}{\sigma_t} - \operatorname{prox}_{\frac{1}{\rho_t} f}\left(\denoiser{x_t}{\sigma_t}\right)\right).
$$

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current noisy iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – Physics operator.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float)) – Diffusion noise standard deviation.
  * **get_model_outputs** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – If `True`, also return the denoised output along with the score. Default: `False`.
* **Returns:**
  DiffPIR gradient, with the same shape and dtype as `x` (and denoised output if `get_model_outputs` is `True`).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

<a id="sphx-glr-backref-deepinv-sampling-diffpirdatafidelity"></a>

## Examples using `DiffPIRDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares three diffusion posterior samplers on a noisy inpainting problem:">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_samplers_thumb.png)

[Posterior sampling with DPS, DiffPIR and DDRM](https://deepinv.org/auto_examples/sampling/demo_diffusion_samplers.md)

  <div class="sphx-glr-thumbnail-title">Posterior sampling with DPS, DiffPIR and DDRM</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares eight approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
