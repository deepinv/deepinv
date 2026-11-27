# Demosaicing

### *class* deepinv.physics.Demosaicing(img_size, pattern='bayer', device='cpu', \*\*kwargs)

Bases: [`Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.md#deepinv.physics.Inpainting)

Demosaicing operator.

The operator keeps one band per pixel, according to a filter array pattern repeated
over the sensor (a Bayer pattern for RGB cameras, or a multispectral filter array
for snapshot multispectral cameras).

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple)) – size of the input images, e.g. (H, W) or (C, H, W). If (H, W), the number of channels C is taken from the pattern.
  * **pattern** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *|* [*list*](https://docs.python.org/3.12/library/stdtypes.html#list) *|* [*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple)) – name of a built-in pattern (see [`PATTERNS`](#deepinv.physics.Demosaicing.PATTERNS)), or a custom 2D tile of band indices.
    `"bayer"` (alias `"bggr"`), `"rggb"`, `"grbg"`, `"gbrg"` are 2x2 Bayer patterns
    (see [https://en.wikipedia.org/wiki/Bayer_filter](https://en.wikipedia.org/wiki/Bayer_filter)), `"brauers"` is a 6-band 3x2 mosaic and
    `"imec_16"` a 16-band 4x4 mosaic. A custom tile must use every band index from 0 to C-1.
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

  16-band snapshot multispectral camera (4x4 mosaic):
  ```pycon
  >>> x = torch.ones(1, 16, 4, 4)
  >>> physics = Demosaicing(img_size=(4, 4), pattern="imec_16")
  >>> physics(x)[0, 0] # band 0 is sampled at the top-left pixel of each tile
  tensor([[1., 0., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., 0., 0.]])
  ```

#### PATTERNS *: [ClassVar](https://docs.python.org/3.12/library/typing.html#typing.ClassVar)[[dict](https://docs.python.org/3.12/library/stdtypes.html#dict)[[str](https://docs.python.org/3.12/library/stdtypes.html#str), [list](https://docs.python.org/3.12/library/stdtypes.html#list)[[list](https://docs.python.org/3.12/library/stdtypes.html#list)[[int](https://docs.python.org/3.12/library/functions.html#int)]]]]* *= {'bayer': [[2, 1], [1, 0]], 'bggr': [[2, 1], [1, 0]], 'brauers': [[0, 1, 2], [3, 4, 5]], 'gbrg': [[1, 2], [0, 1]], 'grbg': [[1, 0], [2, 1]], 'imec_16': [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], 'rggb': [[0, 1], [1, 2]]}*

Built-in patterns. Each tile gives the band index sampled at each pixel of the tile.

#### \_\_mul_\_(other)

Concatenates two forward operators $A = A_1\circ A_2$ via the mul operation.

If the second operator is an [`Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.md#deepinv.physics.Inpainting) (including [`Demosaicing`](#deepinv.physics.Demosaicing)), returns a plain
[`Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.md#deepinv.physics.Inpainting) with the element-wise product mask: the product of two masks is no longer a
filter-array pattern. Otherwise defers to [`Inpainting.__mul__()`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.md#deepinv.physics.Inpainting.__mul__).

* **Parameters:**
  **other** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – Physics operator $A_2$
* **Returns:**
  ([`deepinv.physics.Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) concatenated operator
* **Return type:**
  [*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)

<a id="sphx-glr-backref-deepinv-physics-demosaicing"></a>

## Examples using `Demosaicing`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model terris2025reconstruct to solve inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_foundation_model_thumb.png)

[Inference and fine-tune a foundation model](https://deepinv.org/auto_examples/models/demo_foundation_model.md)

  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_multispectral_demosaicing_thumb.png)

[Multispectral demosaicing from raw sensor data](https://deepinv.org/auto_examples/optimization/demo_multispectral_demosaicing.md)

  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in here.">![](auto_examples/physics/images/thumb/sphx_glr_demo_physics_tour_thumb.png)

[Tour of forward sensing operators](https://deepinv.org/auto_examples/physics/demo_physics_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
