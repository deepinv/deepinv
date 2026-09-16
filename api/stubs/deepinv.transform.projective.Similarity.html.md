# Similarity

### *class* deepinv.transform.projective.Similarity(n_trans=1, theta_max=180.0, theta_z_max=180.0, zoom_factor_min=0.5, shift_max=1.0, skew_max=50.0, x_stretch_factor_min=0.5, y_stretch_factor_min=0.5, padding='reflection', interpolation='bilinear', device='cpu', rng=None)

Bases: [`Homography`](https://deepinv.org/api/stubs/deepinv.transform.Homography.html.md#deepinv.transform.Homography)

Random 2D similarity image transformations using projective transformation framework.

Special case of homography which corresponds to the actions of the similarity subgroup
S(2). Similarity transformations include translations, rotations, reflections and
uniform scale. These transformations are parametrized using geometric parameters in the pinhole camera model. See [`deepinv.transform.Homography`](https://deepinv.org/api/stubs/deepinv.transform.Homography.html.md#deepinv.transform.Homography) for more details.

Generates `n_trans` random transformations concatenated along the batch dimension.

<hr />

* **Example:**
  Apply a random similarity transformation:
  ```pycon
  >>> from deepinv.transform.projective import Similarity
  >>> x = torch.randn(1, 3, 16, 16) # Random 16x16 image
  >>> transform = Similarity(n_trans = 1)
  >>> x_T = transform(x)
  ```
* **Parameters:**
  * **theta_z_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Maximum 2D z-rotation angle in degrees, defaults to 180.
  * **zoom_factor_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Minimum zoom factor (up to 1), defaults to 0.5.
  * **shift_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Maximum shift percentage, where 1 is full shift, defaults to 1.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – kornia padding mode, defaults to “reflection”
  * **interpolation** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – kornia or PIL interpolation mode, defaults to “bilinear”
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, defaults to “cpu”.
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image, defaults to 1.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator, if None, use torch.Generator(), defaults to None

<a id="sphx-glr-backref-deepinv-transform-projective-similarity"></a>

## Examples using `Similarity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div>
<!-- thumbnail-parent-div-close --></div>
