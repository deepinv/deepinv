# MOILoss

### *class* deepinv.loss.MOILoss(physics=None, physics_generator=None, metric=None, apply_noise=True, weight=1.0, rng=None, \*args, \*\*kwargs)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Multi-operator imaging loss

This loss can be used to learn when signals are observed via multiple (possibly incomplete)
forward operators $\{A_g\}_{g=1}^{G}$,
i.e., $y_i = A_{g_i}x_i$ where $g_i\in \{1,\dots,G\}$ (see Tachella *et al.*<sup>[1](#footcite-tachella2022unsupervised)</sup>).

The measurement consistency loss is defined as

$$
\| \hat{x} - \inverse{A_g\hat{x},A_g} \|^2
$$

where $\hat{x}=\inverse{y,A_s}$ is a reconstructed signal (observed via operator $A_s$) and
$A_g$ is a forward operator sampled at random from a set $\{A_g\}_{g=1}^{G}$.

By default, the error is computed using the MSE metric, however any other metric (e.g., $\ell_1$)
can be used as well.

The operators can be passed as a list of physics or as a single physics with a random physics generator.

* **Parameters:**
  * **physics** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) *]* *,* [*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – list of physics containing the $G$ different forward operators
    associated with the measurements, or single physics, or None. If single physics or None, physics generator must be used.
    If None, physics taken during `forward`.
  * **physics_generator** ([*PhysicsGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)) – random physics generator that generates new params, if physics is not a list.
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric used for computing data consistency,
    which is set as the mean squared error by default.
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – total weight of the loss
  * **apply_noise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the augmented measurement is computed with the full sensing model
    $\sensor{\noise{\forw{\hat{x}}}}$ (i.e., noise and sensor model),
    otherwise is generated as $\forw{\hat{x}}$.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – torch random number generator for randomly selecting from physics list. If using physics generator, rng is ignored.

<hr />

* **References:**

* <a id='footcite-tachella2022unsupervised'>**[1]**</a> Julián Tachella, Dongdong Chen, and Mike Davies. Unsupervised learning from incomplete measurements for inverse problems. *Advances in Neural Information Processing Systems*, 35:4983–4995, 2022.

#### forward(x_net, physics, model, \*\*kwargs)

Computes the MOI loss.

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reconstructed image $\inverse{y}$.
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – measurement physics.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

#### next_physics(physics)

Create random physics.

If physics is a list, select one at random. If physics generator is to be used, generate a new set of params at random.

* **Parameters:**
  **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward physics. If None, use physics passed at init.

<a id="sphx-glr-backref-deepinv-loss-moiloss"></a>

## Examples using `MOILoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
