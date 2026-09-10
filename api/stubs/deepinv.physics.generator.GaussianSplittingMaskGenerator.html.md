# GaussianSplittingMaskGenerator

### *class* deepinv.physics.generator.GaussianSplittingMaskGenerator(img_size, split_ratio, pixelwise=True, std_scale=4.0, center_block=(8, 8), device=torch.device('cpu'), rng=None, \*args, \*\*kwargs)

Bases: [`BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator)

Randomly generate Gaussian splitting/inpainting masks.

Generates binary masks with an approximate given split ratio, where samples are weighted according to a spatial Gaussian distribution, where pixels near the center are less likely to be kept.
This mask is used for measurement splitting for MRI in Yaman *et al.*<sup>[1](#footcite-yaman2020self)</sup>.

Can be used either for generating random inpainting masks for [`deepinv.physics.Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.html.md#deepinv.physics.Inpainting), or random splitting masks for [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss).

Optional pass in input_mask to subsample this mask given the split ratio.

Handles both 2D mask (i.e. [C, H, W] from Yaman *et al.*<sup>[1](#footcite-yaman2020self)</sup> and 2D+time dynamic mask (i.e. [C, T, H, W] from Acar *et al.*<sup>[2](#footcite-acar2021self)</sup> generation. Does not handle 1D data (e.g. of shape [C, M])

<hr />

* **Examples:**
  Randomly split input mask using Gaussian weighting
  ```pycon
  >>> from deepinv.physics.generator import GaussianSplittingMaskGenerator
  >>> from deepinv.physics import Inpainting
  >>> physics = Inpainting((1, 3, 3), 0.9)
  >>> gen = GaussianSplittingMaskGenerator((1, 3, 3), split_ratio=0.6, center_block=0)
  >>> gen.step(batch_size=2, input_mask=physics.mask)["mask"].shape
  torch.Size([2, 1, 3, 3])
  ```

See [`deepinv.physics.generator.BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator) for further examples.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the tensor to be masked without batch dimension e.g. of shape (C, H, W) or (C, T, H, W).
    Note this can be overridden on-the-fly by passing in `img_size` or `input_mask` arguments to `step`.
  * **split_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – ratio of values to be kept (i.e. ones).
  * **pixelwise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.
  * **std_scale** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – scale parameter of 2D Gaussian, in pixels.
  * **center_block** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of block in image center that is always kept for MRI autocalibration signal. Either int for square block or 2-tuple (h, w)
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device where the tensor is stored (default: ‘cpu’).
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – the data type of the generated parameters

<hr />

* **References:**

* <a id='footcite-yaman2020self'>**[1]**</a> Burhaneddin Yaman, Seyed Amir Hossein Hosseini, Steen Moeller, Jutta Ellermann, Kâmil Uğurbil, and Mehmet Akçakaya. Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data. *Magnetic resonance in medicine*, 84(6):3172–3191, 2020.
* <a id='footcite-acar2021self'>**[2]**</a> Mert Acar, Tolga Çukur, and İlkay Öksüz. Self-supervised dynamic mri reconstruction. In *Machine Learning for Medical Image Reconstruction: 4th International Workshop, MLMIR 2021, Held in Conjunction with MICCAI 2021, Strasbourg, France, October 1, 2021, Proceedings 4*, 35–44. Springer, 2021.

#### batch_step(input_mask=None, img_size=None)

Create one batch of splitting mask using Gaussian distribution.

Adapted from [https://github.com/byaman14/SSDU/blob/main/masks/ssdu_masks.py](https://github.com/byaman14/SSDU/blob/main/masks/ssdu_masks.py) from SSDU Yaman *et al.*<sup>[1](#footcite-yaman2020self)</sup>.

* **Parameters:**
  * **input_mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – optional mask to be split. If `None`, all pixels are considered. If not `None`, only pixels where `mask==1` are considered. Batch dimension should not be included in shape.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
* **Returns:**
  mask without batch dimension of shape specified either by `img_size`, `input_mask`, or class attribute `img_size`.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

<hr />

* **References:**

#### get_pdf(shape)

Generate a Gaussian distribution.

* **Parameters:**
  **shape** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – (nx, ny) dimensions.
* **Returns:**
  Gaussian Tensor of shape (nx, ny)

<a id="sphx-glr-backref-deepinv-physics-generator-gaussiansplittingmaskgenerator"></a>

## Examples using `GaussianSplittingMaskGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
