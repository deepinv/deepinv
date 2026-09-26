# conv_transpose2d

### deepinv.physics.functional.conv_transpose2d(y, filter, padding='valid', correlation=False)

A helper function performing the 2d transposed convolution 2d of x and filter. The transposed of this operation is [`deepinv.physics.functional.conv2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d.html.md#deepinv.physics.functional.conv2d)

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Image of size `(B, C, W, H)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter of size `(b, c, w, h)` ) where `b` can be either `1` or `B` and `c` can be either `1` or `C`.
  * **correlation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – choose True if you want a cross-correlation (default `False`)
    If `b = 1` or `c = 1`, then this function supports broadcasting as the same as [numpy](https://numpy.org/doc/stable/user/basics.broadcasting.html).
    Otherwise, each channel of each image is convolved with the corresponding kernel.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – options are `'valid'`, `'circular'`, `'replicate'`, `'reflect'`, `'constant'` or `'zeros'`.
    If `padding='valid'` the output is larger than the image (padding) the output has the same size as the image.
    Note that `'constant'` and `'zeros'` are equivalent. Default is `'valid'`.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) : the output
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### NOTE
This functions gives the same result as [`deepinv.physics.functional.conv_transpose2d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose2d_fft.html.md#deepinv.physics.functional.conv_transpose2d_fft). However, for small kernels, this function is faster.
For large kernels, [`deepinv.physics.functional.conv_transpose2d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose2d_fft.html.md#deepinv.physics.functional.conv_transpose2d_fft) is usually faster but requires more memory.
