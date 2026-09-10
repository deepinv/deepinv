# Phase2PhaseSplittingMaskGenerator

### *class* deepinv.physics.generator.Phase2PhaseSplittingMaskGenerator(img_size, device='cpu', rng=None)

Bases: [`BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator)

Phase2Phase splitting mask generator for dynamic data.

To be exclusively used with [`deepinv.loss.mri.Phase2PhaseLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.Phase2PhaseLoss.html.md#deepinv.loss.mri.Phase2PhaseLoss).
Splits dynamic data (i.e. data of shape (B, C, T, H, W)) into even and odd phases in the T dimension.

Used in Eldeniz *et al.*<sup>[1](#footcite-eldeniz2021phase2phase)</sup>.

If input_mask not passed, a blank input mask is used instead.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the tensor to be masked without batch dimension of shape (C, T, H, W).
    Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device where the tensor is stored (default: ‘cpu’).
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – unused.

<hr />

* **References:**

* <a id='footcite-eldeniz2021phase2phase'>**[1]**</a> Cihat Eldeniz, Weijie Gan, Sihao Chen, Tyler J Fraum, Daniel R Ludwig, Yan Yan, Jiaming Liu, Thomas Vahle, Uday Krishnamurthy, Ulugbek S Kamilov, and others. Phase2phase: respiratory motion-resolved reconstruction of free-breathing magnetic resonance imaging using deep learning without a ground truth for improved liver imaging. *Investigative Radiology*, 56(12):809–819, 2021.

#### batch_step(input_mask=None, img_size=None)

Create one batch of splitting mask.

* **Parameters:**
  * **input_mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – optional mask to be split. If `None`, all pixels are considered. If not `None`, only pixels where `mask==1` are considered. Batch dimension should not be included in shape.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
* **Returns:**
  mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)
