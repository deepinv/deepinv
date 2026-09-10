# conv_transpose3d_fft

### deepinv.physics.functional.conv_transpose3d_fft(y, filter, real_fft=True, padding='valid')

A helper function performing the 3d transposed convolution of `y` and `filter` using FFT.

The adjoint of this operation is [`deepinv.physics.functional.conv3d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv3d_fft.html.md#deepinv.physics.functional.conv3d_fft).

If `b = 1` or `c = 1`, then this function applies the same filter for each channel.
Otherwise, each channel of each image is convolved with the corresponding kernel.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Image of size `(B, C, D, H, W)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter of size `(b, c, d, h, w)` where `b` can be either `1` or `B` and `c` can be either `1` or `C`.
  * **real_fft** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – for real filters and images choose `True` (default) to accelerate computation
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – can be `'valid'`, `'circular'`, `'replicate'`, `'reflect'`,  `'constant'` or `'zeros'`.
    If `padding = 'valid'` the output is larger than the image (padding), otherwise the output has the same size as the image.
    Note that `'constant'` and `'zeros'` are equivalent.
    Default is `'valid'`.

#### NOTE
This function and [`deepinv.physics.functional.conv_transpose3d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose3d.html.md#deepinv.physics.functional.conv_transpose3d) are equivalent. However, this function is more efficient for large filters.

* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor): the output of the transposed convolution, which has the same shape as `y` if `padding = 'circular'`, `(B, C, D+d-1, W+w-1, H+h-1)` otherwise.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
