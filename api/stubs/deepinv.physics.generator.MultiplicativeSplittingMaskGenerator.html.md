# MultiplicativeSplittingMaskGenerator

### *class* deepinv.physics.generator.MultiplicativeSplittingMaskGenerator(img_size, split_generator, device=torch.device('cpu'), \*\*kwargs)

Bases: [`BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator)

Multiplicative splitting mask generator.

Randomly generates binary masks using the given `physics_generator`, and multiplies the `input_mask` (i.e. mask that is used to create accelerated measurements).

Given an acceleration mask $M$ sampled from a known distribution, this generator provides masks $M'=M_1 \circ M$ with $M_1$ sampled from `split_generator`,
which is typically the same distribution as $M$.

#### SEE ALSO
[`deepinv.loss.mri.WeightedSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.WeightedSplittingLoss.html.md#deepinv.loss.mri.WeightedSplittingLoss)
: K-weighted splitting loss proposed in Millard and Chiew<sup>[1](#footcite-millard2023theoretical)</sup>,
  where this splitting mask generator is used for self-supervised learning.

<hr />

* **Examples:**
  ```pycon
  >>> from deepinv.physics.generator import GaussianMaskGenerator, MultiplicativeSplittingMaskGenerator
  >>> physics_generator = GaussianMaskGenerator((1, 128, 128), acceleration=4)
  >>> orig_mask = physics_generator.step(batch_size=2)["mask"]
  >>> split_generator = GaussianMaskGenerator((1, 128, 128), acceleration=2)
  >>> mask_generator = MultiplicativeSplittingMaskGenerator((1, 128, 128), split_generator)
  >>> mask_generator.step(batch_size=2, input_mask=orig_mask)["mask"].shape
  torch.Size([2, 1, 128, 128])
  ```

#### NOTE
[`deepinv.physics.generator.MultiplicativeSplittingMaskGenerator`](#deepinv.physics.generator.MultiplicativeSplittingMaskGenerator) calls the `super().step()` function of [`deepinv.physics.generator.BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator) to generate the splitting mask. During initialization, we force `self` to share the same random number generator as `self.split_generator` to correctly propagate seeding to the `self.split_generator` when using `seed` argument in `step`.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the tensor to be masked without batch dimension e.g. of shape (C, H, W) or (C, T, H, W).
    Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
  * **split_generator** ([*deepinv.physics.generator.BaseMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)) – mask generator used for multiplicative splitting
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device where the tensor is stored (default: ‘cpu’).

<hr />

* **References:**

* <a id='footcite-millard2023theoretical'>**[1]**</a> Charles Millard and Mark Chiew. A theoretical framework for self-supervised mr image reconstruction using sub-sampling via variable density noisier2noise. *IEEE transactions on computational imaging*, 9:707–720, 2023.

#### batch_step(input_mask=None, img_size=None)

Create one batch of splitting mask.

* **Parameters:**
  * **input_mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – optional mask to be split. If `None`, all pixels are considered. If not `None`, only pixels where `mask==1` are considered. Batch dimension should not be included in shape.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
* **Returns:**
  mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

<a id="sphx-glr-backref-deepinv-physics-generator-multiplicativesplittingmaskgenerator"></a>

## Examples using `MultiplicativeSplittingMaskGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
