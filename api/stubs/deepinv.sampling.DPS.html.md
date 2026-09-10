# DPS

### *class* deepinv.sampling.DPS(denoiser, schedule='vp', alpha=1.0, num_steps=1000, weight=1.0, verbose=False, device='cpu', dtype=torch.float64, rng=None, \*\*kwargs)

Bases: [`PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion)

Diffusion Posterior Sampling (DPS).

This class implements the Diffusion Posterior Sampling algorithm (DPS) described in Chung *et al.*<sup>[1](#footcite-chung2022diffusion)</sup>.

DPS is an approximation of a gradient-based posterior sampling algorithm,
which has minimal assumptions on the forward model. The only restriction is that
the measurement model has to be differentiable, which is generally the case.

The algorithm solves the reverse-time SDE specified by the `schedule` argument, using the Euler solver, and approximating the conditional score by the DPS data fidelity term, which is defined as follows:

$$
\nabla_{x_t} \log p_t(y|x_t) \approx \nabla_{x_t} \frac{\lambda}{2 \sqrt{m}} \|y - A D_{\sigma_t}(x_t)\|
$$

where $\denoiser{\cdot}{\sigma}$ is a denoising network for noise level $\sigma$, and $\lambda$ is a hyperparameter that controls the weight of the data fidelity term in the approximation of the likelihood gradient.

#### NOTE
This method is a particular instance of the general posterior sampling framework described in [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion), by specifying the data fidelity term as the DPS data fidelity, a SDE and the Euler solver. The user can thus easily modify the algorithm by changing the SDE or the solver, for instance to use a different noise schedule or a different sampling scheme.
Please refer to the example [Building your diffusion posterior sampling method using SDEs](https://deepinv.org/auto_examples/sampling/demo_diffusion_sde.html.md#sphx-glr-auto-examples-sampling-demo-diffusion-sde-py) for a full demonstration of how to modify the algorithm.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser network that can handle different noise levels
  * **schedule** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the noise schedule to use, either `"vp"` (default, which matches the original implementation) for the variance preserving noise schedule, or `"ve"` for the variance exploding noise schedule.
  * **num_steps** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the number of diffusion iterations to run the algorithm (default: 1000)
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – DDIM hyperparameter which controls the stochasticity. Default to 1.0, which corresponds to the original DDPM sampling scheme. Setting it to 0 corresponds to the deterministic DDIM sampling scheme.
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – the weight of the data fidelity term in the approximation of the likelihood gradient. Default to 1.0.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, print the progress of the algorithm
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the device to use for the computations

<hr />

* **References:**

* <a id='footcite-chung2022diffusion'>**[1]**</a> Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In *The Eleventh International Conference on Learning Representations*. 2022.

<a id="sphx-glr-backref-deepinv-sampling-dps"></a>

## Examples using `DPS`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div>
<!-- thumbnail-parent-div-close --></div>
