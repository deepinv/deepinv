# ILVRDataFidelity

### *class* deepinv.sampling.ILVRDataFidelity(gamma=None, weight=1.0, rng=None, \*args, \*\*kwargs)

Bases: [`ScoreSDEDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.ScoreSDEDataFidelity.html.md#deepinv.sampling.ScoreSDEDataFidelity)

Iterative Latent Variable Refinement (ILVR) data-fidelity term.

This corresponds to the $p(y|x_t)$ approximation proposed in
[[26](https://deepinv.org/user_guide/other/biblio.html.md#id36)], and reviewed in [[35](https://deepinv.org/user_guide/other/biblio.html.md#id149)]. ILVR is a
preconditioned version of [`deepinv.sampling.ScoreSDEDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.ScoreSDEDataFidelity.html.md#deepinv.sampling.ScoreSDEDataFidelity): the
mismatch against the noised measurements $y_t = y + \sigma_t\epsilon$ is
lifted back to the image space with the pseudo-inverse
$A^\dagger$ instead of the adjoint $A^\top$,

$$
-\nabla_{x_t} \log p_t(y|x_t) \approx
\lambda \frac{A^\dagger \left(A x_t - y_t\right)}{\sigma_y^2 + \gamma_t^2},
\qquad A^\dagger = \left(A^\top A\right)^{-1} A^\top,
$$

where $\lambda$, exposed as `weight`, controls the scale of the
data-fidelity term.

* **Parameters:**
  * **gamma** (*Callable* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – annealing parameter $\gamma_t$. If `None`
    (default), $\gamma_t = \sigma_t$, the current diffusion noise level.
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weighting factor $\lambda$. Default: `1.0`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – Random number generator used to noise the measurements,
    for reproducibility. Default: `None`.

#### grad(x, y, physics, sigma, \*args, \*\*kwargs)

Compute the ILVR data-fidelity gradient
$\lambda A^\dagger \left(A x_t - y_t\right) / (\sigma_y^2 + \gamma_t^2)$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current noisy iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – Diffusion noise standard deviation.
* **Returns:**
  ILVR gradient, with the same shape as `x`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-sampling-ilvrdatafidelity"></a>

## Examples using `ILVRDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
