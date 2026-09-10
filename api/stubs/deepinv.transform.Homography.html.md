# Homography

### *class* deepinv.transform.Homography(n_trans=1, theta_max=180.0, theta_z_max=180.0, zoom_factor_min=0.5, shift_max=1.0, skew_max=50.0, x_stretch_factor_min=0.5, y_stretch_factor_min=0.5, padding='reflection', interpolation='bilinear', device='cpu', rng=None)

Bases: [`Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)

Random projective transformations (homographies).

The homography is parameterized by
geometric parameters. By fixing these parameters, subgroup transformations are
retrieved, see Wang and Davies<sup>[1](#footcite-wang2024perspective)</sup>.

For example, setting x_stretch_factor_min = y_stretch_factor_min = zoom_factor_min = 1,
theta_max = theta_z_max = skew_max = 0 gives a pure translation.

Subgroup transformations include [`deepinv.transform.projective.Affine`](https://deepinv.org/api/stubs/deepinv.transform.projective.Affine.html.md#deepinv.transform.projective.Affine), [`deepinv.transform.projective.Similarity`](https://deepinv.org/api/stubs/deepinv.transform.projective.Similarity.html.md#deepinv.transform.projective.Similarity),
[`deepinv.transform.projective.Euclidean`](https://deepinv.org/api/stubs/deepinv.transform.projective.Euclidean.html.md#deepinv.transform.projective.Euclidean) along with the basic [`deepinv.transform.Shift`](https://deepinv.org/api/stubs/deepinv.transform.Shift.html.md#deepinv.transform.Shift),
[`deepinv.transform.Rotate`](https://deepinv.org/api/stubs/deepinv.transform.Rotate.html.md#deepinv.transform.Rotate) and semigroup [`deepinv.transform.Scale`](https://deepinv.org/api/stubs/deepinv.transform.Scale.html.md#deepinv.transform.Scale).

Transformations with perspective effects (i.e. pan+tilt) are recovered by setting
theta_max > 0.

Generates `n_trans` random transformations concatenated along the batch dimension.

<hr />

* **Example:**
  Apply a random projective transformation:
  ```pycon
  >>> from deepinv.transform.projective import Homography
  >>> x = torch.randn(1, 3, 16, 16) # Random 16x16 image
  >>> transform = Homography(n_trans = 1)
  >>> x_T = transform(x)
  ```
* **Parameters:**
  * **theta_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Maximum pan+tilt angle in degrees, defaults to 180.
  * **theta_z_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Maximum 2D z-rotation angle in degrees, defaults to 180.
  * **zoom_factor_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Minimum zoom factor (up to 1), defaults to 0.5.
  * **shift_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Maximum shift percentage, where 1 is full shift, defaults to 1.
  * **skew_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Maximum skew parameter, defaults to 50.
  * **x_stretch_factor_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Min stretch factor along the x-axis (up to 1), defaults to 0.5.
  * **y_stretch_factor_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Min stretch factor along the y-axis (up to 1), defaults to 0.5.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – kornia padding mode, defaults to “reflection”
  * **interpolation** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – kornia or PIL interpolation mode, defaults to “bilinear”
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, defaults to “cpu”.
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image, defaults to 1.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator, if None, use torch.Generator(), defaults to None

<hr />

* **References:**

* <a id='footcite-wang2024perspective'>**[1]**</a> Andrew Wang and Mike Davies. Perspective-equivariant imaging: an unsupervised framework for multispectral pansharpening. *arXiv e-prints*, pages arXiv–2403, 2024.

<a id="sphx-glr-backref-deepinv-transform-homography"></a>

## Examples using `Homography`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
