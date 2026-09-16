# BaseMaskGenerator

### *class* deepinv.physics.generator.BaseMaskGenerator(img_size, acceleration=4, center_fraction=None, rng=None, device='cpu', \*args, \*\*kwargs)

Bases: [`PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator), [`ABC`](https://docs.python.org/3.9/library/abc.html#abc.ABC)

Base generator for MRI acceleration masks.

Generate a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and undersampling in the high frequencies.

The type of undersampling is determined by the child class. The mask is repeated across channels and randomly varies across batch dimension.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – image size, either (H, W) or (C, H, W) or (C, T, H, W), where optional C is channels, and optional T is number of time-steps
  * **acceleration** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – acceleration factor, defaults to 4
  * **center_fraction** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – fraction of lines to sample in low frequencies (center of k-space). If 0, there is no fixed low-freq sampling. Defaults to None.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – torch random generator. Defaults to None.

#### calculate_lines(W)

Calculate number of lines and center lines from total width.

* **Parameters:**
  **W** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – total number of columns (width of mask)

#### *abstractmethod* get_pdf()

Get mask probability density function (PDF).

* **Return torch.Tensor:**
  unnormalized 1D vector representing pdf evaluated across mask columns.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *abstractmethod* sample_mask(mask)

Given empty mask, sample lines according to child class sampling strategy.

This must be implemented in child classes. Time dimension is specified but can be ignored if needed.

* **Parameters:**
  **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – empty mask of shape (B, C, T, H, W)
* **Return torch.Tensor:**
  sampled mask of shape (B, C, T, H, W)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### step(batch_size=1, seed=None, img_size=None, \*\*kwargs)

Create a mask of vertical lines.

* **Parameters:**
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batch_size.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – optional: the seed for the random number generator, to reseed on-the-fly.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – if not `None`, generate masks of this 2D image shape and override `img_size` attribute, must be of form `(H, W)`.
* **Returns:**
  dictionary with key **‘mask’**: tensor of size (batch_size, C, H, W) or (batch_size, C, T, H, W) with values in {0, 1}.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

<a id="sphx-glr-backref-deepinv-physics-generator-basemaskgenerator"></a>

## Examples using `BaseMaskGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
