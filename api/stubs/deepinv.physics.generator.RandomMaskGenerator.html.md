# RandomMaskGenerator

### *class* deepinv.physics.generator.RandomMaskGenerator(img_size, acceleration=4, center_fraction=None, rng=None, device='cpu', \*args, \*\*kwargs)

Bases: [`BaseMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)

Generator for MRI Cartesian acceleration masks using random uniform undersampling.

Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and random uniform undersampling in the high frequencies.

Supports k-t sampling, where the mask is selected randomly across time.

The mask is repeated across channels and randomly varies across batch dimension.

For parameter descriptions see [`deepinv.physics.generator.mri.BaseMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)

<hr />

* **Examples:**
  Random k-t mask generator for a 8x64x64 video:
  ```pycon
  >>> from deepinv.physics.generator import RandomMaskGenerator
  >>> generator = RandomMaskGenerator((2, 8, 64, 64), acceleration=8, center_fraction=0.04) # C, T, H, W
  >>> params = generator.step(batch_size=1)
  >>> mask = params["mask"]
  >>> mask.shape
  torch.Size([1, 2, 8, 64, 64])
  ```

#### get_pdf(W)

Create one-dimensional uniform probability density function across columns, ignoring any fixed sampling columns.

* **Parameters:**
  **W** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – total number of columns (width of mask)
* **Return torch.Tensor:**
  unnormalized 1D vector representing pdf evaluated across mask columns.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-physics-generator-randommaskgenerator"></a>

## Examples using `RandomMaskGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
