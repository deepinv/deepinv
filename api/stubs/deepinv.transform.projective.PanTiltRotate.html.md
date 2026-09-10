# PanTiltRotate

### *class* deepinv.transform.projective.PanTiltRotate(n_trans=1, theta_max=180.0, theta_z_max=180.0, zoom_factor_min=0.5, shift_max=1.0, skew_max=50.0, x_stretch_factor_min=0.5, y_stretch_factor_min=0.5, padding='reflection', interpolation='bilinear', device='cpu', rng=None)

Bases: [`Homography`](https://deepinv.org/api/stubs/deepinv.transform.Homography.html.md#deepinv.transform.Homography)

Random 3D camera rotation image transformations using projective transformation framework.

Special case of homography which corresponds to the actions of the 3D camera rotation,
or “pan+tilt+rotate” subgroup from Wang and Davies<sup>[1](#footcite-wang2024perspective)</sup>.

The transformations simulate panning, tilting or rotating the camera, leading to a
“perspective” effect. The subgroup is isomorphic to SO(3).

See [`deepinv.transform.Homography`](https://deepinv.org/api/stubs/deepinv.transform.Homography.html.md#deepinv.transform.Homography) for more details.

Generates `n_trans` random transformations concatenated along the batch dimension.

<hr />

* **Example:**
  Apply a random pan+tilt+rotate transformation:
  ```pycon
  >>> from deepinv.transform.projective import PanTiltRotate
  >>> x = torch.randn(1, 3, 16, 16) # Random 16x16 image
  >>> transform = PanTiltRotate(n_trans = 1)
  >>> x_T = transform(x)
  ```
* **Parameters:**
  * **theta_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Maximum pan+tilt angle in degrees, defaults to 180.
  * **theta_z_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Maximum 2D z-rotation angle in degrees, defaults to 180.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – kornia padding mode, defaults to “reflection”
  * **interpolation** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – kornia or PIL interpolation mode, defaults to “bilinear”
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, defaults to “cpu”.
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image, defaults to 1.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator, if None, use torch.Generator(), defaults to None

<hr />

* **References:**

* <a id='footcite-wang2024perspective'>**[1]**</a> Andrew Wang and Mike Davies. Perspective-equivariant imaging: an unsupervised framework for multispectral pansharpening. *arXiv e-prints*, pages arXiv–2403, 2024.

<a id="sphx-glr-backref-deepinv-transform-projective-pantiltrotate"></a>

## Examples using `PanTiltRotate`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
