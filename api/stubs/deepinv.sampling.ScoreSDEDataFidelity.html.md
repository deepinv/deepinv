# ScoreSDEDataFidelity

### *class* deepinv.sampling.ScoreSDEDataFidelity(gamma=None, weight=1.0, rng=None, \*args, \*\*kwargs)

Bases: [`ALDDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.ALDDataFidelity.html.md#deepinv.sampling.ALDDataFidelity)

Score-SDE data-fidelity term.

This corresponds to the $p(y|x_t)$ approximation proposed in
[[154](https://deepinv.org/user_guide/other/biblio.html.md#id111)], and reviewed in [[37](https://deepinv.org/user_guide/other/biblio.html.md#id152)]. The difference with
[`deepinv.sampling.ALDDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.ALDDataFidelity.html.md#deepinv.sampling.ALDDataFidelity) is that the measurements are noised to the
current diffusion noise level before the mismatch is computed,

$$
y_t = y + \sigma_t\epsilon, \qquad \epsilon\sim\mathcal{N}(0,\mathrm{Id}),
$$

so that $y_t$ and $A x_t$ live at the same noise level. The resulting
negative log-likelihood gradient is

$$
-\nabla_{x_t} \log p_t(y|x_t) \approx
\lambda \frac{A^\top \left(A x_t - y_t\right)}{\sigma_y^2 + \gamma_t^2},
$$

where $\lambda$, exposed as `weight`, controls the scale of the
data-fidelity term.

#### NOTE
[[37](https://deepinv.org/user_guide/other/biblio.html.md#id152)] writes this approximation without a guidance strength,
noting that it then differs from [`deepinv.sampling.ALDDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.ALDDataFidelity.html.md#deepinv.sampling.ALDDataFidelity) only
by the noising of the measurements. We keep the annealed guidance strength
$\sigma_y^2 + \gamma_t^2$ here, so that the term stays balanced against
the unconditional score across noise levels.

* **Parameters:**
  * **gamma** (*Callable* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – annealing parameter $\gamma_t$. If `None`
    (default), $\gamma_t = \sigma_t$, the current diffusion noise level.
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weighting factor $\lambda$. Default: `1.0`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – Random number generator used to noise the measurements,
    for reproducibility. Default: `None`.

#### forward(\*args, \*\*kwargs)

Not implemented: the measurements are re-noised at every call, so this term has
no deterministic value, see [`grad()`](#deepinv.sampling.ScoreSDEDataFidelity.grad).

#### grad(x, y, physics, sigma, \*args, \*\*kwargs)

Compute the Score-SDE data-fidelity gradient
$\lambda A^\top \left(A x_t - y_t\right) / (\sigma_y^2 + \gamma_t^2)$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current noisy iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – Diffusion noise standard deviation.
* **Returns:**
  Score-SDE gradient, with the same shape as `x`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-sampling-scoresdedatafidelity"></a>

## Examples using `ScoreSDEDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.html.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
