# conv_transpose3d

### deepinv.physics.functional.conv_transpose3d(y, filter, padding='valid', correlation=False)

A helper function to perform 3D transpose convolution.
The transposed of this operation is [`deepinv.physics.functional.conv3d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv3d.html.md#deepinv.physics.functional.conv3d).

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Image of size `(B, C, D, H, W)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter of size `(b, c, d, h, w)` where `b` can be either `1` or `B` and `c` can be either `1` or `C`.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – can be `'valid'` (default), `'circular'`, `'replicate'`, `'reflect'`, `'constant'` or `'zeros'`.
    If `padding = 'valid'` the output is larger than the image (padding), otherwise the output has the same size as the image.
    Note that `'constant'` and `'zeros'` are equivalent. Default is `'valid'`.
  * **correlation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – choose `True` if you want the transpose of the cross-correlation (default `False`).
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor): the output of the convolution, which has the shape `(B, C, D+d-1, W+w-1, H+h-1)` if `padding = 'valid'` and the same shape as `y` otherwise.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
