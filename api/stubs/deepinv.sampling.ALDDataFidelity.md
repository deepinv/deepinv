# ALDDataFidelity

### *class* deepinv.sampling.ALDDataFidelity(gamma=None, weight=1.0, \*args, \*\*kwargs)

Bases: [`NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.md#deepinv.sampling.NoisyDataFidelity)

Score-based annealed Langevin dynamics (Score-ALD) data-fidelity term.

This corresponds to the $p(y|x_t)$ approximation proposed in
[[69](https://deepinv.org/user_guide/other/biblio.md#id153)], and reviewed in [[37](https://deepinv.org/user_guide/other/biblio.md#id152)], given by

$$
p_t(y|x_t) \approx \mathcal{N} \left( y; A x_t,
\left(\sigma_y^2 + \gamma_t^2\right)\mathrm{Id} \right).
$$

The resulting negative log-likelihood gradient is

$$
-\nabla_{x_t} \log p_t(y|x_t) \approx
\lambda \frac{A^\top \left(A x_t - y\right)}{\sigma_y^2 + \gamma_t^2},
$$

where $\sigma_y$ is the measurement noise level and $\lambda$,
exposed as `weight`, controls the scale of the data-fidelity term.

#### NOTE
$\gamma_t$ should decrease along the diffusion, so that the guidance
strengthens as $x_t$ gets closer to the data manifold. The default
`gamma=None` follows [[69](https://deepinv.org/user_guide/other/biblio.md#id153)] and uses the current diffusion
noise level, $\gamma_t=\sigma_t$.

* **Parameters:**
  * **gamma** (*Callable* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – annealing parameter $\gamma_t$. If `None`
    (default), $\gamma_t = \sigma_t$, the current diffusion noise level. A
    `float` uses a constant value, and a `Callable` is evaluated as
    $\gamma_t = \text{gamma}(\sigma_t)$.
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weighting factor $\lambda$. Default: `1.0`.

#### forward(x, y, physics, sigma, \*args, \*\*kwargs)

Returns the loss term
$\lambda \| A x_t - y \|^2 / \left(2(\sigma_y^2 + \gamma_t^2)\right)$,
whose gradient is given by [`grad()`](#deepinv.sampling.ALDDataFidelity.grad).

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – forward operator.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – Diffusion noise standard deviation.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss term, of size `B` the batch size.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, y, physics, sigma, \*args, \*\*kwargs)

Compute the Score-ALD data-fidelity gradient.

$$
-\nabla_{x_t} \log p_t(y|x_t) \approx
\lambda \frac{A^\top \left(A x_t - y\right)}{\sigma_y^2 + \gamma_t^2}.
$$

The measurement noise level $\sigma_y$ is read from `physics.noise_model.sigma` when the noise is Gaussian, and is taken to be zero otherwise.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Current noisy iterate.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – physics model.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – Diffusion noise standard deviation.
* **Returns:**
  Score-ALD gradient, with the same shape as `x`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-sampling-alddatafidelity"></a>

## Examples using `ALDDataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">![](auto_examples/sampling/images/thumb/sphx_glr_demo_flow_matching_thumb.png)

[Flow-Matching for posterior sampling and unconditional generation](https://deepinv.org/auto_examples/sampling/demo_flow_matching.md)

  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div>
<!-- thumbnail-parent-div-close --></div>
