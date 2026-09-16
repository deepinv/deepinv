# MomentMatchingDataFidelity

### *class* deepinv.sampling.MomentMatchingDataFidelity(denoiser=None, weight=1.0, clip=None, cg_max_iter=3, cg_tol=1e-4, verbose=False)

Bases: [`NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.html.md#deepinv.sampling.NoisyDataFidelity)

Moment-matching data-fidelity term for diffusion posterior sampling.

This corresponds to the $p(y|x_t)$ approximation proposed in [[132](https://deepinv.org/user_guide/other/biblio.html.md#id151)].
For the VE parametrization, Moment Matching approximates the full conditional distribution with the
Gaussian with mean and covariance given by the denoiser and its Jacobian:

$$
p(x_0|x_t) \approx \mathcal{N} \left(
x_0;D(x_t,\sigma_t),\Sigma_t(x_t) \right),
\qquad
\Sigma_t(x_t)=\sigma_t^2J_D(x_t,\sigma_t).
$$

The resulting negative log-likelihood gradient is

$$
-\nabla_{x_t} \log p_t(y|x_t) \approx \lambda J_D(x_t, \sigma_t)^\top A^\top
\left(\sigma_t^2 A J_D(x_t, \sigma_t) A^\top
+ \sigma_y^2\mathrm{Id}\right)^{-1}
\left(A D(x_t, \sigma_t) - y\right).
$$

The parameter $\lambda$, exposed as `weight`, controls the scale of
the data-fidelity term. The Jacobian products are evaluated with
vector-Jacobian products, without
materializing the denoiser Jacobian, and the measurement-space system is
approximated with conjugate gradient.
The measurement noise level $\sigma_y$ is read from
`physics.noise_model.sigma`.

#### NOTE
Conjugate gradient assumes that the effective moment-matching operator
is symmetric positive definite, as is expected for an exact MMSE
denoiser covariance.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – Denoiser network. It may be left as
    `None` when the data fidelity is passed to
    [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion), which supplies its denoiser.
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weighting factor $\lambda$. Default: `1.0`.
  * **clip** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
  * **cg_max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Maximum number of conjugate-gradient iterations.
    Default: `3`.
  * **cg_tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Relative conjugate-gradient tolerance. Default: `1e-4`.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, print conjugate-gradient convergence
    information. Default: `False`.

#### grad(x, y, physics, sigma, \*args, get_model_outputs=False, \*\*kwargs)

Compute the moment-matching data-fidelity gradient.

> $$
> -\nabla_{x_t} \log p_t(y|x_t) \approx \lambda J_D(x_t, \sigma_t)^\top A^\top
> \left(\sigma_t^2 A J_D(x_t, \sigma_t) A^\top
> + \sigma_y^2\mathrm{Id}\right)^{-1}
> \left(A D(x_t, \sigma_t) - y\right).
> $$
* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current noisy iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Linear physics operator.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – Diffusion noise standard deviation.
  * **get_model_outputs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, also return the denoised output along with the score. Default to `False`.
* **Returns:**
  Moment-matching gradient, with the same shape and dtype as
  `x`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

<a id="sphx-glr-backref-deepinv-sampling-momentmatchingdatafidelity"></a>

## Examples using `MomentMatchingDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
