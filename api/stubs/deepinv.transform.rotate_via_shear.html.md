# rotate_via_shear

### *class* deepinv.transform.rotate_via_shear(image, angle, center=None)

Bases:

2D rotation of an image by shear composition through FFT.

#### Deprecated
Deprecated since version Use: [`deepinv.transform.RotateViaShear`](https://deepinv.org/api/stubs/deepinv.transform.RotateViaShear.html.md#deepinv.transform.RotateViaShear) instead.

* **Parameters:**
  * **image** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of shape `(B, C, H, W)`.
  * **angle** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.12/library/functions.html#float) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int)) – rotation angles in degrees.
  * **center** ([*tuple*](https://docs.python.org/3.12/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.12/library/functions.html#int) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int) *]* *,* *None*) – center of rotation.
* **Returns:**
  rotated images.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
