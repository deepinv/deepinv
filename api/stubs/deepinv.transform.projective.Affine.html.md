# Affine

### *class* deepinv.transform.projective.Affine(n_trans=1, theta_max=180.0, theta_z_max=180.0, zoom_factor_min=0.5, shift_max=1.0, skew_max=50.0, x_stretch_factor_min=0.5, y_stretch_factor_min=0.5, padding='reflection', interpolation='bilinear', device='cpu', rng=None)

Bases: [`Homography`](https://deepinv.org/api/stubs/deepinv.transform.Homography.html.md#deepinv.transform.Homography)

Random affine image transformations using projective transformation framework.

Special case of homography which corresponds to the actions of the affine subgroup
Aff(3). Affine transformations include translations, rotations, reflections,
skews, and stretches. These transformations are parametrized using geometric parameters in the pinhole camera model.
See [`deepinv.transform.Homography`](https://deepinv.org/api/stubs/deepinv.transform.Homography.html.md#deepinv.transform.Homography) for more details.

Generates `n_trans` random transformations concatenated along the batch dimension.

<hr />

* **Example:**
  Apply a random affine transformation:
  ```pycon
  >>> from deepinv.transform.projective import Affine
  >>> x = torch.randn(1, 3, 16, 16) # Random 16x16 image
  >>> transform = Affine(n_trans = 1)
  >>> x_T = transform(x)
  ```
* **Parameters:**
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

<a id="sphx-glr-backref-deepinv-transform-projective-affine"></a>

## Examples using `Affine`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
