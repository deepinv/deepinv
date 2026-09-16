# conv_transpose2d_fft

### deepinv.physics.functional.conv_transpose2d_fft(y, filter, real_fft=True, padding='valid')

A helper function performing the 2d transposed convolution 2d of `x` and `filter` using FFT.
The adjoint of this operation is [`deepinv.physics.functional.conv2d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d_fft.html.md#deepinv.physics.functional.conv2d_fft).

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Image of size `(B, C, W, H)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter of size `(b, c, w, h)` ) where `b` can be either `1` or `B` and `c` can be either `1` or `C`.
    If `b = 1` or `c = 1`, then this function supports broadcasting as the same as [numpy](https://numpy.org/doc/stable/user/basics.broadcasting.html). Otherwise, each channel of each image is convolved with the corresponding kernel.
  * **real_fft** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – for real filters and images choose `True` (default) to accelerate computation.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – can be `'valid'`, `'circular'`, `'replicate'`, `'reflect'`,  `'constant'` or `'zeros'`.
    If `padding = 'valid'` the output is larger than the image (padding), otherwise the output has the same size as the image.
    Note that `'constant'` and `'zeros'` are equivalent.
    Default is `'valid'`.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor): the output of the convolution, which has the same shape as $y$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### NOTE
This functions gives the same result as [`deepinv.physics.functional.conv_transpose2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose2d.html.md#deepinv.physics.functional.conv_transpose2d).
However, for large kernels, this function is faster but requires more memory.
For small kernels, consider using [`deepinv.physics.functional.conv_transpose2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose2d.html.md#deepinv.physics.functional.conv_transpose2d).
