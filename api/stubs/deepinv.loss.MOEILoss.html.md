# MOEILoss

### *class* deepinv.loss.MOEILoss(transform, physics=None, physics_generator=None, metric=None, apply_noise=True, weight=1.0, no_grad=False, rng=None)

Bases: [`EILoss`](https://deepinv.org/api/stubs/deepinv.loss.EILoss.html.md#deepinv.loss.EILoss), [`MOILoss`](https://deepinv.org/api/stubs/deepinv.loss.MOILoss.html.md#deepinv.loss.MOILoss)

Multi-operator equivariant imaging.

This loss extends the equivariant loss [`deepinv.loss.EILoss`](https://deepinv.org/api/stubs/deepinv.loss.EILoss.html.md#deepinv.loss.EILoss), where the signals are not only
assumed to be invariant to a group of transformations, but also observed
via multiple (possibly incomplete) forward operators $\{A_s\}_{s=1}^{S}$,
i.e., $y_i = A_{s_i}x_i$ where $s_i\in \{1,\dots,S\}$.

The multi-operator equivariance loss is defined as

$$
\| T_g \hat{x} - \inverse{A_2 T_g \hat{x}, A_2}\|^2
$$

where $\hat{x}=\inverse{y,A_1}$ is a reconstructed signal (observed via operator $A_1$),
$A_2$ is a forward operator sampled at random from a set $\{A_2\}_{s=1}^{S}$ and
$T_g$ is a transformation sampled at random from a group $g\sim\group$.

By default, the error is computed using the MSE metric, however any other metric (e.g., $\ell_1$)
can be used as well.

The operators can be passed as a list of physics or as a single physics with a random physics generator.

See [`deepinv.loss.EILoss`](https://deepinv.org/api/stubs/deepinv.loss.EILoss.html.md#deepinv.loss.EILoss) for all parameter details for EI.

* **Parameters:**
  * **transform** ([*deepinv.transform.Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)) – Transform to generate the virtually
    augmented measurement. It can be any torch-differentiable function (e.g., a `torch.nn.Module`).
  * **physics** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) *]* *,* [*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – list of physics containing the $G$ different forward operators
    associated with the measurements, or single physics, or None. If single physics or None, physics generator must be used.
    If None, physics taken during `forward`.
  * **physics_generator** ([*PhysicsGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)) – random physics generator that generates new params, if physics is not a list.
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Metric used to compute the error between the reconstructed augmented measurement and the reference
    image.
  * **apply_noise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the augmented measurement is computed with the full sensing model
    $\sensor{\noise{\forw{\hat{x}}}}$ (i.e., noise and sensor model),
    otherwise is generated as $\forw{\hat{x}}$.
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weight of the loss.
  * **no_grad** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the gradient does not propagate through $T_g$. Default: `False`.
    This option is useful for super-resolution problems, see Scanvic *et al.*<sup>[1](#footcite-scanvic2026scale)</sup>.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – torch random number generator for randomly selecting from physics list. If using physics generator, rng is ignored.

<hr />

* **References:**

* <a id='footcite-scanvic2026scale'>**[1]**</a> Jérémy Scanvic, Mike Davies, Patrice Abry, and Julián Tachella. Scale-equivariant imaging: self-supervised learning for image super-resolution and deblurring. *IEEE Transactions on Computational Imaging*, 2026.

#### forward(x_net, physics, model, \*\*kwargs)

Computes the MO-EI loss

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reconstructed image $\inverse{y}$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss.
