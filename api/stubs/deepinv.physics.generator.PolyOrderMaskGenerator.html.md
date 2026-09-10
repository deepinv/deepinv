# PolyOrderMaskGenerator

### *class* deepinv.physics.generator.PolyOrderMaskGenerator(\*args, poly_order=8, \*\*kwargs)

Bases: [`BaseMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)

Generator for MRI Cartesian acceleration masks using polynomial variable density.

Generates a mask of vertical lines for MRI acceleration with fixed sampling in low frequencies (center of k-space) and polynomial order sampling in the high frequencies.

This is achieved by the following:

1. Create 1D polynomial function $(1 - r)^{p}$ where $r$ is the distance from the center and $p$ is the polynomial order.
2. Scale so that its mean matches desired acceleration factor
3. Use the function as a probability density function (pdf) to sample from a Bernoulli.

The mask is repeated across channels and randomly varies across batch dimension.

Algorithm taken from Millard and Chiew<sup>[1](#footcite-millard2023theoretical)</sup>.

* **Examples:**
  ```pycon
  >>> from deepinv.physics.generator.mri import PolyOrderMaskGenerator
  >>> generator = PolyOrderMaskGenerator((2, 128, 128), acceleration=8, center_fraction=0.04, poly_order=8)
  >>> params = generator.step(batch_size=1)
  >>> mask = params["mask"]
  >>> mask.shape
  torch.Size([1, 2, 128, 128])
  ```

For other parameter descriptions see [`deepinv.physics.generator.mri.BaseMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)

* **Parameters:**
  **poly_order** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – polynomial order of the sampling pdf

<hr />

* **References:**

* <a id='footcite-millard2023theoretical'>**[1]**</a> Charles Millard and Mark Chiew. A theoretical framework for self-supervised mr image reconstruction using sub-sampling via variable density noisier2noise. *IEEE transactions on computational imaging*, 9:707–720, 2023.
