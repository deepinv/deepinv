# DDRMDataFidelity

### *class* deepinv.sampling.DDRMDataFidelity(denoiser=None, weight=1.0, eta=0.85, clip=None, sigma=0.01, eps=1e-6)

Bases: [`NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.md#deepinv.sampling.NoisyDataFidelity)

Denoising Diffusion Restoration Models (DDRM) data-fidelity term.

It can be used to build custom DDRM-based diffusion samplers with [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.md#deepinv.sampling.PosteriorDiffusion). A self-contained implementation of DDRM is available in [`deepinv.sampling.DDRM`](https://deepinv.org/api/stubs/deepinv.sampling.DDRM.md#deepinv.sampling.DDRM).

This adapts the spectral measurement update of Kawar *et al.*<sup>[1](#footcite-kawar2022denoising)</sup>
for a decomposable linear operator $A = U \Sigma V^\top$
([`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.md#deepinv.physics.DecomposablePhysics)) and Gaussian measurement noise with
standard deviation $\sigma_y$, in the VE parametrization $x_t = x_0 + \sigma_t \omega$.

In the SVD domain, let $\bar{x}_i = (V^\top x_t)_i$, $\bar{y}_i = (U^\top y)_i / s_i$
and $\nu_i = \sigma_y / s_i$ for each singular value $s_i$.
Components with $s_i < \sigma_y$ are treated as unmeasured and left unchanged.

It provides an approximation of the gradient of the log-likelihood $\log p_t(y \vert x_t)$ in the SVD domain as:

$$
-\nabla_{x_t} \log p_t(y \vert x_t) \approx
\lambda V \Sigma^\top W \left(\sigma_y^2 \mathrm{Id} + \sigma_t^2 \Sigma \Sigma^\top\right)^{-1}
\left(\Sigma V^\top \denoiser{\tilde{x}_t}{\sigma_t} - U^\top y\right),
$$

where $W$ is diagonal with $W_i = 1$ if $\sigma_t \geq \nu_i$
and $W_i = \left(\sigma_t / (2 \nu_i)\right)^2$ otherwise; and
$\tilde{x}_t = V \left(\bar{x} + \lambda \kappa \odot (\bar{y} - \bar{x})\right)$, with

$$
\kappa_i = \begin{cases}
1 - \frac{1}{\sqrt{2}} \sqrt{1 - \nu_i^2 / \sigma_t^2} & \text{if } \sigma_t \geq \nu_i, \\
\min\left(\sqrt{1 - \eta^2} \, \sigma_t / \nu_i, 1\right) & \text{if } \sigma_t < \nu_i.
\end{cases}
$$

Here $\eta$ is the stochasticity parameter of DDRM, and $\lambda$, exposed as `weight`
(corresponding to $\eta_b$ in Kawar *et al.*<sup>[1](#footcite-kawar2022denoising)</sup>), controls the strength of both steps.
The measurement noise level $\sigma_y$ is read from `physics.noise_model.sigma`
when available, and defaults to `sigma` otherwise.

The approximation $-\nabla_{x_t} \log p_t(y \vert x_t)$ also carries measurement noise through $U^\top y$; as $y$ is fixed, the same noise realization enters the estimate on every step.
The parameters $\eta$ controls the amount of additional noise added at each step.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.md#deepinv.models.Denoiser)) – Denoiser network. It may be left as `None`
  * **weight** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Weighting factor $\lambda$ (called $\eta_b$ in Kawar *et al.*<sup>[1](#footcite-kawar2022denoising)</sup>). Default: `1.0`.
  * **eta** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Stochasticity parameter $\eta \in [0, 1]$ of DDRM. Default: `0.85`.
  * **clip** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.12/library/functions.html#float) *]*) – If not `None`, clip the denoised output into `[clip[0], clip[1]]`
    interval. Default: `None`.
  * **sigma** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Fallback measurement noise standard deviation $\sigma_y$ used when
    `physics.noise_model` does not define `sigma`. Default: `0.01`.
  * **eps** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – Numerical floor added to the spectral denominator. Default: `1e-6`.

<hr />

* **References:**

* <a id='footcite-kawar2022denoising'>**[1]**</a> Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. *Advances in Neural Information Processing Systems*, 35:23593–23606, 2022.

#### grad(x, y, physics, sigma, \*args, get_model_outputs=False, \*\*kwargs)

Compute the DDRM noisy data-fidelity gradient approximation.

$$
-\nabla_{x_t} \log p_t(y \vert x_t) \approx
\lambda V \Sigma^\top W \left(\sigma_y^2 \mathrm{Id} + \sigma_t^2 \Sigma \Sigma^\top\right)^{-1}
\left(\Sigma V^\top \denoiser{\tilde{x}_t}{\sigma_t} - U^\top y\right)
$$

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current noisy iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.DecomposablePhysics*](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.md#deepinv.physics.DecomposablePhysics)) – Decomposable physics operator.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float)) – Diffusion noise standard deviation.
  * **get_model_outputs** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – If `True`, also return the denoised output $\denoiser{\tilde{x}_t}{\sigma_t}$
    along with the score. Default: `False`.
* **Returns:**
  DDRM gradient, with the same shape and dtype as `x` (and denoised output if `get_model_outputs` is `True`).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.12/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

<a id="sphx-glr-backref-deepinv-sampling-ddrmdatafidelity"></a>

## Examples using `DDRMDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares three diffusion posterior samplers on a noisy inpainting problem:">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_samplers_thumb.png)

[Posterior sampling with DPS, DiffPIR and DDRM](https://deepinv.org/auto_examples/sampling/demo_diffusion_samplers.md)

  <div class="sphx-glr-thumbnail-title">Posterior sampling with DPS, DiffPIR and DDRM</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares eight approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
