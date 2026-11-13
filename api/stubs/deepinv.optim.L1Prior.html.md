# L1Prior

### *class* deepinv.optim.L1Prior(\*args, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

$\ell_1$ prior $\reg{x} = \| x \|_1$.

#### fn(x, \*args, \*\*kwargs)

Computes the regularizer $\reg{x} = \| x \|_1$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the prior is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) prior $\reg{x}$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, \*args, ths=1.0, gamma=1.0, \*\*kwargs)

Calculates the proximity operator of the l1 regularization term $\regname$ at $x$.

More precisely, it computes

$$
\operatorname{prox}_{\gamma g}(x) = \operatorname{sign}(x) \max(|x| - \gamma, 0)

$$

where $\gamma$ is a stepsize.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **gamma** ([*float*](https://docs.python.org/3.12/library/functions.html#float)) – stepsize of the proximity operator.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator at $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-l1prior"></a>

## Examples using `L1Prior`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the zea ultrasound toolbox stevens2026zea with deepinv to form ultrasound images from raw channel data, by posing the image formation step, known as beamforming, as an inverse problem. We use the differentiable ultrasound processing pipeline of zea to compute the forward operator A of the linear model that relates the tissue reflectivity to the channel data, and solve the inverse problem with deepinv.optim.FISTA. First, a quick refresher on ultrasound image formation.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_zea_ultrasound_thumb.png)

[Ultrasound regularized beamforming with zea toolbox](https://deepinv.org/auto_examples/external-libraries/demo_zea_ultrasound.html.md)

  <div class="sphx-glr-thumbnail-title">Ultrasound regularized beamforming with zea toolbox</div>
</div>
<!-- thumbnail-parent-div-close --></div>
