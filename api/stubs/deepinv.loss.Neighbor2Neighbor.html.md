# Neighbor2Neighbor

### *class* deepinv.loss.Neighbor2Neighbor(metric=None, gamma=2.0)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Neighbor2Neighbor loss.

Implements the self-supervised Neighbor2Neighbor loss Huang *et al.*<sup>[1](#footcite-huang2021neighbor2neighbor)</sup>.

Splits the noisy measurements using two masks $A_1$ and $A_2$, each choosing a different neighboring
map (see details in Huang *et al.*<sup>[1](#footcite-huang2021neighbor2neighbor)</sup>). The self-supervised loss is computed as:

$$
\| A_2 y - R(A_1 y)\|^2 + \gamma \| A_2 y - R(A_1 y) - (A_2 R(y) - A_1 R(y))\|^2
$$

where $R$ is the trainable denoiser network, $\gamma>0$ is a regularization parameter
and no gradient is propagated when computing $R(y)$.

By default, the error is computed using the MSE metric, however any other metric (e.g., $\ell_1$)
can be used as well.

The code has been adapted from the repository [https://github.com/TaoHuang2018/Neighbor2Neighbor](https://github.com/TaoHuang2018/Neighbor2Neighbor).

* **Parameters:**
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric used for computing data consistency,
    which is set as the mean squared error by default.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – regularization parameter $\gamma$.

<hr />

* **References:**

* <a id='footcite-huang2021neighbor2neighbor'>**[1]**</a> Tao Huang, Songjiang Li, Xu Jia, Huchuan Lu, and Jianzhuang Liu. Neighbor2neighbor: self-supervised denoising from single noisy images. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 14781–14790. 2021.

#### forward(y, physics, model, \*\*kwargs)

Computes the neighbor2neighbor loss.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-neighbor2neighbor"></a>

## Examples using `Neighbor2Neighbor`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
