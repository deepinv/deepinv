# TimeMixin

### *class* deepinv.utils.TimeMixin

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Base class for temporal capabilities for physics and models.

Implements various methods to add or remove the time dimension.

Also provides template methods for temporal physics to implement.

#### *static* average(x, mask=None, dim=2)

Flatten time dim of x by averaging across frames.

If mask is non-overlapping in time dim, then this will simply be the sum across frames.

* **Parameters:**
  * **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape `(B,C,T,H,W)` (e.g. time-varying k-space)
  * **mask** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – mask showing where `x` is non-zero. If not provided, then calculated from `x`.
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – time dimension, defaults to 2 (i.e. shape `B,C,T,H,W`)
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

#### *static* flatten_C(x)

Flatten time dim into channel dim.

Use when channel dim doesn’t matter and you don’t want to deal with annoying batch dimension problems (e.g. for transforms).

* **Parameters:**
  **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B, C, T, H, W)
* **Returns:**
  output tensor of shape (B, C\*T, H, W)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* repeat(x, target, dim=2)

Repeat static image across new time dim T times. Opposite of `average`.

* **Parameters:**
  * **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape `(B,C,H,W)`
  * **target** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – any tensor of desired shape `(B,C,T,H,W)`
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – time dimension, defaults to 2 (i.e. shape `B,C,T,H,W`)
* **Returns:**
  tensor with new time dim of shape `(B,C,T,H,W)`
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* unflatten(x, batch_size=1)

Creates new time dim from batch dim. Opposite of `flatten`.

* **Parameters:**
  * **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (B\*T, C, H, W)
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batch size, defaults to 1
* **Returns:**
  output tensor of shape (B, C, T, H, W)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* wrap_flatten_C(f)

Flatten time dim into channel dim, apply function, then unwrap.

The first argument is assumed to be the tensor to be flattened.

* **Parameters:**
  **f** ([*Callable*](https://docs.python.org/3.9/library/typing.html#typing.Callable) *[* *[*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *,* [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – function to be wrapped
* **Returns:**
  wrapped function
* **Return type:**
  [*Callable*](https://docs.python.org/3.9/library/typing.html#typing.Callable)

<a id="sphx-glr-backref-deepinv-utils-timemixin"></a>

## Examples using `TimeMixin`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
