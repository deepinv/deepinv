# Demosaicing

### *class* deepinv.physics.Demosaicing(img_size, pattern='bayer', device='cpu', \*\*kwargs)

Bases: [`Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.html.md#deepinv.physics.Inpainting)

Demosaicing operator.

The operator chooses one color per pixel according to the pattern specified.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – size of the input images, e.g. (H, W) or (C, H, W).
  * **pattern** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – `bayer` (see [https://en.wikipedia.org/wiki/Bayer_filter](https://en.wikipedia.org/wiki/Bayer_filter)) or other patterns.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – `gpu` or `cpu`

<hr />

* **Examples:**
  Demosaicing operator using Bayer pattern for a 4x4 image:
  ```pycon
  >>> from deepinv.physics import Demosaicing
  >>> x = torch.ones(1, 3, 4, 4)
  >>> physics = Demosaicing(img_size=(4, 4))
  >>> physics(x)[0, 1, :, :] # Green channel
  tensor([[0., 1., 0., 1.],
          [1., 0., 1., 0.],
          [0., 1., 0., 1.],
          [1., 0., 1., 0.]])
  ```

<a id="sphx-glr-backref-deepinv-physics-demosaicing"></a>

## Examples using `Demosaicing`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
