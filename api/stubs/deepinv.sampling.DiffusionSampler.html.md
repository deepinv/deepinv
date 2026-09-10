# DiffusionSampler

### *class* deepinv.sampling.DiffusionSampler(diffusion, max_iter=1e2, clip=(-1, 2), thres_conv=1e-1, g_statistic=lambda x: ..., verbose=True, save_chain=False)

Bases: [`BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling)

Turns a diffusion method into a Monte Carlo sampler.

Unlike diffusion methods, the resulting sampler computes the mean and variance of the distribution
by running the diffusion multiple times.

See the docs for [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling) for more information. It uses the helper class [`deepinv.sampling.DiffusionIterator`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionIterator.html.md#deepinv.sampling.DiffusionIterator).

* **Parameters:**
  * **diffusion** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – a diffusion model
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the number of samples to generate
  * **clip** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – the clip range
  * **g_statistic** (*Callable*) – the algorithm computes mean and variance of the g function, by default $g(x) = x$.
  * **thres_conv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the convergence threshold for the mean and variance
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to print the progress
  * **save_chain** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to save the chain
  * **thinning** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the thinning factor
  * **burnin_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the burnin ratio

#### forward(y, physics, seed=None)

Runs the diffusion model to obtain the posterior mean and variance of the reconstruction of the measurements y.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements
  * **seed** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Random seed for generating the samples
* **Returns:**
  (tuple of torch.Tensor) containing the posterior mean and variance.

<a id="sphx-glr-backref-deepinv-sampling-diffusionsampler"></a>

## Examples using `DiffusionSampler`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm :footcitekawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div>
<!-- thumbnail-parent-div-close --></div>
