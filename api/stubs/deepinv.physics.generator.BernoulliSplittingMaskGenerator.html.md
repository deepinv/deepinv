# BernoulliSplittingMaskGenerator

### *class* deepinv.physics.generator.BernoulliSplittingMaskGenerator(img_size, split_ratio, pixelwise=True, random_split_ratio=False, min_split_ratio=0.0, max_split_ratio=1.0, device=torch.device('cpu'), dtype=torch.float32, rng=None, \*args, \*\*kwargs)

Bases: [`PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)

Base generator for splitting/inpainting masks.

Generates binary masks with an approximate given split ratio, according to a Bernoulli distribution. Can be used either for generating random inpainting masks for [`deepinv.physics.Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.html.md#deepinv.physics.Inpainting), or random splitting masks for [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss).

Optional pass in input_mask to subsample this mask given the split ratio. For mask ratio to be almost exactly as specified, use this option with a flat mask of ones as input.

<hr />

* **Examples:**
  Generate random mask
  ```pycon
  >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
  >>> gen = BernoulliSplittingMaskGenerator((1, 3, 3), split_ratio=0.6)
  >>> gen.step(batch_size=2)["mask"].shape
  torch.Size([2, 1, 3, 3])
  ```

  Generate splitting mask from given input_mask
  ```pycon
  >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
  >>> from deepinv.physics import Inpainting
  >>> physics = Inpainting((1, 100, 100), 0.9)
  >>> gen = BernoulliSplittingMaskGenerator((1, 100, 100), split_ratio=0.6)
  >>> gen.step(batch_size=2, input_mask=physics.mask)["mask"].shape
  torch.Size([2, 1, 100, 100])
  ```

  Generate splitting mask from given `input_mask` with random split ratio for each sample in the batch
  ```pycon
  >>> gen = BernoulliSplittingMaskGenerator((1, 100, 100), split_ratio=0.6, random_split_ratio=True, min_split_ratio=0.1, max_split_ratio=0.9)
  >>> mask = gen.step(batch_size=2, input_mask=physics.mask, seed=10)["mask"]
  >>> (mask[0] == 0).sum()/mask[0].numel()  # 0.1 < split_ratio < 0.9
  tensor(0.5782)
  >>> (mask[1] == 0).sum()/mask[1].numel()  # 0.1 < split_ratio < 0.9
  tensor(0.2905)
  ```

  Generate splitting mask with new 2D shape than that given at initialization
  ```pycon
  >>> gen.step(img_size=(71, 73))["mask"].shape
  torch.Size([1, 1, 71, 73])
  ```
* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the tensor to be masked without batch dimension e.g. of shape (C, H, W) or (C, M) or (M,).
    Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
  * **split_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – ratio of values to be kept.
  * **pixelwise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.
  * **random_split_ratio** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, `split_ratio` is randomly sampled from `[min_split_ratio, max_split_ratio]` at each step.
  * **min_split_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – minimum split ratio. Only used if `random_split_ratio` is True.
  * **max_split_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum split ratio. Only used if `random_split_ratio` is True.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device where the tensor is stored (default: ‘cpu’).
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – the data type of the generated parameters
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – torch random number generator.

#### batch_step(input_mask=None, img_size=None)

Create one batch of splitting mask.

* **Parameters:**
  * **input_mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – optional mask to be split. If `None`, all pixels are considered. If not `None`, only pixels where `mask==1` are considered. Batch dimension should not be included in shape.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
* **Returns:**
  mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

#### check_pixelwise(input_mask=None)

Check if pixelwise can be used given input_mask dimensions and img_size dimensions

#### step(batch_size=1, input_mask=None, img_size=None, seed=None, \*\*kwargs)

Generate a random mask.

If `input_mask` is None, generates a standard random mask that can be used for [`deepinv.physics.Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.html.md#deepinv.physics.Inpainting).
If `input_mask` is specified, splits the input mask into subsets given the split ratio.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batch_size. If None, no batch dimension is created. If input_mask passed and has its own batch dimension > 1, batch_size is ignored.
  * **input_mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – optional mask to be split. If None, all pixels are considered. If not None, only pixels where mask==1 are considered. input_mask shape can optionally include a batch dimension.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.
* **Returns:**
  dictionary with key **‘mask’**: tensor of size `(batch_size, *img_size)` with values in {0, 1}.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

<a id="sphx-glr-backref-deepinv-physics-generator-bernoullisplittingmaskgenerator"></a>

## Examples using `BernoulliSplittingMaskGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
