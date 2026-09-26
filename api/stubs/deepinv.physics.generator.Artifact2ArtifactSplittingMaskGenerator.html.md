# Artifact2ArtifactSplittingMaskGenerator

### *class* deepinv.physics.generator.Artifact2ArtifactSplittingMaskGenerator(img_size, split_size=2, device='cpu', rng=None)

Bases: [`Phase2PhaseSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.Phase2PhaseSplittingMaskGenerator.html.md#deepinv.physics.generator.Phase2PhaseSplittingMaskGenerator)

Artifact2Artifact splitting mask generator for dynamic data.

To be exclusively used with [`deepinv.loss.mri.Artifact2ArtifactLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.Artifact2ArtifactLoss.html.md#deepinv.loss.mri.Artifact2ArtifactLoss).
Randomly selects a chunk from dynamic data (i.e. data of shape (B, C, T, H, W)) in the T dimension and puts zeros in the rest of the mask.

Artifact2Artifact was introduced by Liu *et al.*<sup>[1](#footcite-liu2020rare)</sup>.

If input_mask not passed, a blank input mask is used instead.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the tensor to be masked without batch dimension of shape (C, T, H, W).
    Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
  * **split_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – time-length of chunk. Must divide `img_size[1]` exactly. If `tuple`, one is randomly selected each time.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device where the tensor is stored (default: ‘cpu’).
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – torch random number generator.

<hr />

* **References:**

* <a id='footcite-liu2020rare'>**[1]**</a> Jiaming Liu, Yu Sun, Cihat Eldeniz, Weijie Gan, Hongyu An, and Ulugbek S Kamilov. Rare: image reconstruction using deep priors learned without groundtruth. *IEEE Journal of Selected Topics in Signal Processing*, 14(6):1088–1099, 2020.

#### batch_step(input_mask=None, img_size=None, persist_prev=False)

Create one batch of splitting mask.

* **Parameters:**
  * **input_mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – optional mask to be split. If `None`, all pixels are considered. If not `None`, only pixels where `mask==1` are considered. Batch dimension should not be included in shape.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
  * **persist_prev** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, the selected chunk will be different from the previous time it was called. This is used so input chunk is compared to a different output chunk. Default to `False`.
* **Returns:**
  mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)
