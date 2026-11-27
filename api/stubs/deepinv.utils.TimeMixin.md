# TimeMixin

### *class* deepinv.utils.TimeMixin

Bases: [`object`](https://docs.python.org/3.12/library/functions.html#object)

Base class for temporal capabilities for physics and models.

Implements various methods to add or remove the time dimension.

#### *static* average(x, mask=None, dim=2)

Flatten time dim of x by averaging across frames.

If mask is non-overlapping in time dim, then this will simply be the sum across frames.

* **Parameters:**
  * **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape `(B,C,T,H,W)` (e.g. time-varying k-space)
  * **mask** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – mask showing where `x` is non-zero. If not provided, then calculated from `x`.
  * **dim** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – time dimension, defaults to 2 (i.e. shape `B,C,T,H,W`)
* **Returns:**
  flattened tensor with time dim removed of shape `(B,C,H,W)`
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* flatten(x)

Flatten time dim into batch dim.

Lets non-dynamic algorithms process dynamic data by treating time frames as batches.

* **Parameters:**
  **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B, C, T, H, W)
* **Returns:**
  output tensor of shape (B\*T, C, H, W)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* repeat(x, target, dim=2)

Repeat static image across new time dim T times. Opposite of `average`.

* **Parameters:**
  * **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape `(B,C,H,W)`
  * **target** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – any tensor of desired shape `(B,C,T,H,W)`
  * **dim** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – time dimension, defaults to 2 (i.e. shape `B,C,T,H,W`)
* **Returns:**
  tensor with new time dim of shape `(B,C,T,H,W)`
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* unflatten(x, batch_size=1)

Creates new time dim from batch dim. Opposite of `flatten`.

* **Parameters:**
  * **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B\*T, C, H, W)
  * **batch_size** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – batch size, defaults to 1
* **Returns:**
  output tensor of shape (B, C, T, H, W)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-utils-timemixin"></a>

## Examples using `TimeMixin`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Many imaging modalities are not acquired all at once but sequentially in &quot;shots&quot;, e.g. kspace lines in MRI or projections in CT. Since the time between shots is non-negligible, the subject may move during scanning, and reconstructions that assume a static subject will be motion-corrupted. Modelling a transform per shot allows us to simulate this motion corruption and, given the motion parameters, to compensate for it during reconstruction.">![](auto_examples/physics/images/thumb/sphx_glr_demo_motion_mri_thumb.png)

[Motion simulation and motion-compensated reconstruction](https://deepinv.org/auto_examples/physics/demo_motion_mri.md)

  <div class="sphx-glr-thumbnail-title">Motion simulation and motion-compensated reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">![](auto_examples/physics/images/thumb/sphx_glr_demo_mri_tour_thumb.png)

[Tour of MRI functionality in DeepInverse](https://deepinv.org/auto_examples/physics/demo_mri_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_artifact2artifact_thumb.png)

[Self-supervised MRI reconstruction with Artifact2Artifact](https://deepinv.org/auto_examples/self-supervised-learning/demo_artifact2artifact.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div>
<!-- thumbnail-parent-div-close --></div>
