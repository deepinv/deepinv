# PiGDMDataFidelity

### *class* deepinv.sampling.PiGDMDataFidelity(denoiser=None, weight=1.0, clip=None, cg_max_iter=3, cg_tol=1e-4, verbose=False)

Bases: [`NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.md#deepinv.sampling.NoisyDataFidelity)

Pseudoinverse-guided diffusion model (PiGDM) data-fidelity term.

This corresponds to the $p(y|x_t)$ approximation proposed in [[153](https://deepinv.org/user_guide/other/biblio.md#id151)].
For the VE parametrization $x_t=x_0+\sigma_t\omega$, PiGDM uses the
isotropic Gaussian approximation

$$
p(x_0|x_t) \approx \mathcal{N} \left( x_0;D(x_t,\sigma_t),\Sigma_t(x_t) \right),
\qquad \Sigma_t(x_t)=r_t^2\mathrm{Id},
\qquad r_t^2=\frac{\sigma_t^2}{1+\sigma_t^2}.
$$

For a linear forward operator and Gaussian measurement noise with standard
deviation $\sigma_y$,
integrating this approximation gives a Gaussian approximation of
$p_t(y|x_t)$. Its negative log-likelihood gradient is

$$
-\nabla_{x_t} \log p_t(y|x_t) \approx \lambda J_D(x_t, \sigma_t)^\top A^\top
 \left(r_t^2 A A^\top + \sigma_y^2\mathrm{Id}\right)^{-1}
 \left(A D(x_t, \sigma_t) - y\right).
$$

Here $D$ is a denoiser and $J_D$ is its Jacobian. The parameter
$\lambda$, exposed as `weight`, controls the scale of the
data-fidelity term. The inverse is evaluated
exactly for [`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.md#deepinv.physics.DecomposablePhysics) operators and
approximated with conjugate gradient for other linear operators.
The measurement noise level $\sigma_y$ is read from
`physics.noise_model.sigma`.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.md#deepinv.models.Denoiser)) – Denoiser network. It may be left as
    `None` when the data fidelity is passed to
    [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.md#deepinv.sampling.PosteriorDiffusion), which supplies its denoiser.
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weighting factor $\lambda$. Default: `1.0`.
  * **clip** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
  * **cg_max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Maximum number of conjugate-gradient iterations.
    Default: `3`.
  * **cg_tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Relative conjugate-gradient tolerance. Default: `1e-4`.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, print conjugate-gradient convergence
    information. Default: `False`.

#### grad(x, y, physics, sigma, \*args, get_model_outputs=False, \*\*kwargs)

Compute the PiGDM data-fidelity gradient.

$$
-\nabla_{x_t} \log p_t(y|x_t) \approx \lambda J_D(x_t, \sigma_t)^\top A^\top
 \left(r_t^2 A A^\top + \sigma_y^2\mathrm{Id}\right)^{-1}
 \left(A D(x_t, \sigma_t) - y\right).
$$

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current noisy iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – Linear physics operator.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – Diffusion noise standard deviation.
  * **get_model_outputs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, also return the denoised output along with the score. Default to `False`.
* **Returns:**
  PiGDM gradient, with the same shape and dtype as `x`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### solve_inverse(physics, u, r_t2, sigma_y)

Apply
$(r_t^2 A A^\top + \sigma_y^2\mathrm{Id})^{-1}$ to `u`.

* **Parameters:**
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – Linear physics operator.
  * **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor in the measurement space.
  * **r_t2** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – PiGDM covariance parameter
    $r_t^2$.
  * **sigma_y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – Measurement noise standard deviation
    $\sigma_y$.
* **Returns:**
  Solution in the measurement space.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-sampling-pigdmdatafidelity"></a>

## Examples using `PiGDMDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
