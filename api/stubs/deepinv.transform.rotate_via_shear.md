# rotate_via_shear

### *class* deepinv.transform.rotate_via_shear(image, angle, center=None)

Bases:

2D rotation of image by angle via shear composition through FFT.

* **Parameters:**
  * **image** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of shape `(B,C,H,W)`
  * **angle** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – input rotation angles in degrees of shape `(B,)`
* **Returns:**
  torch.Tensor containing the rotated images of shape `(B, C, H, W )`
