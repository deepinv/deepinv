# conv_filter_transpose2d_fft

### deepinv.physics.functional.conv_filter_transpose2d_fft(x, y, filter_size, real_fft=True, padding='circular', correlation=False)

Apply the adjoint of 2D convolution with respect to its filter using FFTs.

This function gives the same result as
[`deepinv.physics.functional.conv_filter_transpose2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_filter_transpose2d.md#deepinv.physics.functional.conv_filter_transpose2d). It is generally
faster for large filter supports, while the non-fft version is faster for
small filters.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor of shape `(B, C, H, W)`.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Adjoint input. Its spatial shape must match the output
    of [`deepinv.physics.functional.conv2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d.md#deepinv.physics.functional.conv2d) applied to `x`.
  * **filter_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Filter size `(H_f, W_f)`.
  * **real_fft** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Use real FFTs for real-valued inputs.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – One of `"valid"`, `"circular"`, `"replicate"`,
    `"reflect"`, `"constant"` or `"zeros"`.
  * **correlation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, return the filter adjoint of
    cross-correlation rather than convolution.
* **Returns:**
  Per-channel filter adjoint of shape `(B, C, H_f, W_f)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
