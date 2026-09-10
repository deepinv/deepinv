# conv2d

### deepinv.physics.functional.conv2d(x, filter, padding='valid', correlation=False)

A helper function performing the 2d convolution of images `x` and `filter`.

The adjoint of this operation is [`deepinv.physics.functional.conv_transpose2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose2d.html.md#deepinv.physics.functional.conv_transpose2d)

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Image of size `(B, C, W, H)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter of size `(b, c, w, h)` where `b` can be either `1` or `B`
    and `c` can be either `1` or `C`.
    Filter center is at `(hh, ww)` where `hh = h//2` if h is odd and
    `hh = h//2 - 1` if h is even. Same for `ww`.
  * **correlation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – choose True if you want a cross-correlation (default `False`)

If `b = 1` or `c = 1`, then this function supports broadcasting as the same as [numpy](https://numpy.org/doc/stable/user/basics.broadcasting.html). Otherwise, each channel of each image is convolved with the corresponding kernel.

* **Parameters:**
  **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – (options = `'valid'`, `'circular'`, `'replicate'`, `'reflect'`, `'constant'` or `'zeros'`). If `padding = 'valid'` the output is smaller than the image (no padding), otherwise the output has the same size as the image. Note that `'constant'` and `'zeros'` are equivalent. Default is `'valid'`.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor): the blurry output.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### NOTE
Contrary to PyTorch’s [`torch.nn.functional.conv2d()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d), which performs a cross-correlation, this function performs a convolution by default unless `correlation=True`.

This function gives the same result as [`deepinv.physics.functional.conv2d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d_fft.html.md#deepinv.physics.functional.conv2d_fft). However, for small kernels, this function is faster.
For large kernels, [`deepinv.physics.functional.conv2d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d_fft.html.md#deepinv.physics.functional.conv2d_fft) is usually faster but requires more memory.

## Examples using `conv2d`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div>
<!-- thumbnail-parent-div-close --></div>
