# EquispacedMaskGenerator

### *class* deepinv.physics.generator.EquispacedMaskGenerator(img_size, acceleration=4, center_fraction=None, rng=None, device='cpu', \*args, \*\*kwargs)

Bases: [`BaseMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)

Generator for MRI Cartesian acceleration masks using uniform (equispaced) non-random undersampling with random offset.

Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and equispaced undersampling in the high frequencies.

The number of lines selected with equal spacing are at a proportion that reaches the desired acceleration rate taking into consideration the number of low-freq lines, so that the total number of lines is (N / acceleration).

Supports k-t sampling, where the uniform mask is sheared across time.

The mask is repeated across channels and the offset varies randomly across batch dimension. Based off fastMRI code [https://github.com/facebookresearch/fastMRI](https://github.com/facebookresearch/fastMRI)

For parameter descriptions see [`deepinv.physics.generator.mri.BaseMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)

<hr />

* **Examples:**
  Equispaced k-t mask generator for a 8x64x64 video:
  ```pycon
  >>> from deepinv.physics.generator import EquispacedMaskGenerator
  >>> generator = EquispacedMaskGenerator((2, 8, 64, 64), acceleration=8, center_fraction=0.04) # C, T, H, W
  >>> params = generator.step(batch_size=1)
  >>> mask = params["mask"]
  >>> mask.shape
  torch.Size([1, 2, 8, 64, 64])
  ```

<a id="sphx-glr-backref-deepinv-physics-generator-equispacedmaskgenerator"></a>

## Examples using `EquispacedMaskGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
