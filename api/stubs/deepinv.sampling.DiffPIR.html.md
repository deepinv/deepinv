# DiffPIR

### *class* deepinv.sampling.DiffPIR(model, data_fidelity, sigma=0.05, max_iter=100, zeta=0.1, lambda_=7.0, verbose=False, device='cpu')

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Diffusion PnP Image Restoration (DiffPIR).

This class implements the Diffusion PnP image restoration algorithm (DiffPIR) described in Zhu *et al.*<sup>[1](#footcite-zhu2023denoising)</sup>.

The DiffPIR algorithm is inspired on a half-quadratic splitting (HQS) plug-and-play algorithm, where the denoiser
is a conditional diffusion denoiser, combined with a diffusion process. The algorithm writes as follows,
for $t$ decreasing from $T$ to $1$:

> $$
> x_{0}^{t} &= D_{\theta}(x_t, \frac{\sqrt{1-\overline{\alpha}_t}}{\sqrt{\overline{\alpha}_t}}) \\
> \widehat{x}_{0}^{t} &= \operatorname{prox}_{2 f(y, \cdot) /{\rho_t}}(x_{0}^{t}) \\
> \widehat{\varepsilon} &= \left(x_t - \sqrt{\overline{\alpha}_t} \,\,
> \widehat{x}_{0}^t\right)/\sqrt{1-\overline{\alpha}_t} \\
> \varepsilon_t &= \mathcal{N}(0, \mathbf{I}) \\
> x_{t-1} &= \sqrt{\overline{\alpha}_t} \,\, \widehat{x}_{0}^t + \sqrt{1-\overline{\alpha}_t}
> \left(\sqrt{1-\zeta} \,\, \widehat{\varepsilon} + \sqrt{\zeta} \,\, \varepsilon_t\right)

> $$

where $D_\theta(\cdot,\sigma)$ is a Gaussian denoiser network with noise level $\sigma$
and $f(y, \cdot)$ is the data fidelity
term.

#### NOTE
The algorithm might require careful tunning of the hyperparameters $\lambda$ and $\zeta$ to
obtain optimal results.

* **Parameters:**
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – a conditional noise estimation model
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the noise level of the data
  * **data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)) – the data fidelity operator
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the number of iterations to run the algorithm (default: 100)
  * **zeta** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – hyperparameter $\zeta$ for the sampling step (must be between 0 and 1). Default: 1.0.
  * **lambda** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – hyperparameter $\lambda$ for the data fidelity step
    ($\rho_t = \lambda \frac{\sigma_n^2}{\bar{\sigma}_t^2}$ in the paper where the optimal value range
    between 3.0 and 25.0 depending on the problem). Default: `7.0`.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, print progress
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the device to use for the computations

<hr />

* **Examples:**
  Denoising diffusion restoration model using a pretrained DRUNet denoiser:

```default
import deepinv as dinv
device = dinv.utils.get_device(verbose=False)
x = 0.5 * torch.ones(1, 3, 32, 32, device=device) # Define a plain gray 32x32 image
physics = dinv.physics.Inpainting(mask=0.5, img_size=(3, 32, 32),
   noise_model=dinv.physics.GaussianNoise(0.1), device=device)
y = physics(x) # Measurements
denoiser = dinv.models.DRUNet(device=device)
model = dinv.sampling.DiffPIR(model=denoiser, data_fidelity=dinv.optim.data_fidelity.L2(),
   device=device) # Define the DiffPIR model
xhat = model(y, physics) # Run the DiffPIR algorithm
print((dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x))) # should be True
```

<hr />

* **References:**

* <a id='footcite-zhu2023denoising'>**[1]**</a> Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 1219–1229. 2023.

#### compute_alpha(betas, t)

Compute the alpha sequence from the beta sequence.

#### find_nearest(array, value)

Find the argmin of the nearest value in an array.

#### forward(y, physics, seed=None, x_init=None)

Runs the diffusion to obtain a random sample of the posterior distribution.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the measurements.
  * **physics** ([*deepinv.physics.LinearPhysics*](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)) – the physics operator.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the noise level of the data.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.
  * **x_init** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the initial guess for the reconstruction.

#### get_alpha_beta()

Get the alpha and beta sequences for the algorithm. This is necessary for mapping noise levels to timesteps.

#### get_alpha_prod(beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=1000)

Get the alpha sequences; this is necessary for mapping noise levels to timesteps when performing pure denoising.

#### get_noise_schedule(sigma)

Get the noise schedule for the algorithm.

<a id="sphx-glr-backref-deepinv-sampling-diffpir"></a>

## Examples using `DiffPIR`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from :footcitezhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div>
<!-- thumbnail-parent-div-close --></div>
