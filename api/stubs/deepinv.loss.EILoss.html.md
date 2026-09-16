# EILoss

### *class* deepinv.loss.EILoss(transform, metric=None, apply_noise=True, weight=1.0, no_grad=False, \*args, \*\*kwargs)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Equivariant imaging self-supervised loss.

Assumes that the set of signals is invariant to a group of transformations (rotations, translations, etc.)
in order to learn from incomplete measurement data alone.
The EI loss, as proposed by Chen *et al.*<sup>[1](#footcite-chen2021equivariant)</sup>, is defined as

$$
\| T_g \hat{x} - \inverse{\forw{T_g \hat{x}}}\|^2
$$

where $\hat{x}=\inverse{y}$ is a reconstructed signal and
$T_g$ is a transformation sampled at random from a group $g\sim\group$.

By default, the error is computed using the MSE metric, however any other metric (e.g., $\ell_1$)
can be used as well.

* **Parameters:**
  * **transform** ([*deepinv.transform.Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)) – Transform to generate the virtually augmented measurement.
    It can be any torch-differentiable function (e.g., a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module))
    including [torchvision transforms](https://pytorch.org/vision/stable/transforms.html).
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Metric used to compute the error between the reconstructed augmented measurement and the reference
    image.
  * **apply_noise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the augmented measurement is computed with the full sensing model
    $\sensor{\noise{\forw{\hat{x}}}}$ (i.e., noise and sensor model),
    otherwise is generated as $\forw{\hat{x}}$.
  * **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Weight of the loss.
  * **no_grad** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the gradient does not propagate through $T_g$. Default: `False`.
    This option is useful for super-resolution problems, see Scanvic *et al.*<sup>[2](#footcite-scanvic2026scale)</sup> for details.

<hr />

* **References:**

* <a id='footcite-chen2021equivariant'>**[1]**</a> Dongdong Chen, Julián Tachella, and Mike E Davies. Equivariant imaging: learning beyond the range space. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 4379–4388. 2021.
* <a id='footcite-scanvic2026scale'>**[2]**</a> Jérémy Scanvic, Mike Davies, Patrice Abry, and Julián Tachella. Scale-equivariant imaging: self-supervised learning for image super-resolution and deblurring. *IEEE Transactions on Computational Imaging*, 2026.

#### forward(x_net, physics, model, \*\*kwargs)

Computes the EI loss

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reconstructed image $\inverse{y}$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-eiloss"></a>

## Examples using `EILoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
