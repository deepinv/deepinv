# conv3d

### deepinv.physics.functional.conv3d(x, filter, padding='valid', correlation=False)

A helper function to perform 3D convolution of images `x` and `filter`.

The transposed of this operation is [`deepinv.physics.functional.conv_transpose3d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose3d.html.md#deepinv.physics.functional.conv_transpose3d).

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Image of size `(B, C, D, H, W)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter of size `(b, c, d, h, w)` where `b` can be either `1` or `B` and `c` can be either `1` or `C`.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – can be `'valid'` (default), `'circular'`, `'replicate'`, `'reflect'`,  `'constant'` or `'zeros'`.
    If `padding = 'valid'` the output is smaller than the image (no padding), otherwise the output has the same size as the image.
    Note that `'constant'` and `'zeros'` are equivalent. Default is `'valid'`.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor): the output of the convolution, which has the shape `(B, C, D-d+1, W-w+1, H-h+1)` if `padding = 'valid'` and the same shape as `x` otherwise.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### NOTE
Contrary to Pytorch’s [`torch.nn.functional.conv3d()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html#torch.nn.functional.conv3d), which performs a cross-correlation, this function performs a convolution by default unless `correlation=True`.
