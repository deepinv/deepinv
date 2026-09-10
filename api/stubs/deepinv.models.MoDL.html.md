# MoDL

### *class* deepinv.models.MoDL(denoiser=None, num_iter=3)

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)

MoDL unfolded network.

The model is a simple unrolled network using half-quadratic splitting
where the prox is replaced by a trainable denoising prior.

This was proposed for MRI reconstruction in Aggarwal *et al.*<sup>[1](#footcite-aggarwal2018modl)</sup>.

* **Parameters:**
  * **denoiser** ([*Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – backbone denoiser model. If `None`, uses [`deepinv.models.DnCNN`](https://deepinv.org/api/stubs/deepinv.models.DnCNN.html.md#deepinv.models.DnCNN)
  * **num_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of unfolded layers (“cascades”), defaults to 3.

<hr />

* **References:**

* <a id='footcite-aggarwal2018modl'>**[1]**</a> Hemant K Aggarwal, Merry P Mani, and Mathews Jacob. Modl: model-based deep learning architecture for inverse problems. *IEEE transactions on medical imaging*, 38(2):394–405, 2018.

<a id="sphx-glr-backref-deepinv-models-modl"></a>

## Examples using `MoDL`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
