# Upsampling

### *class* deepinv.physics.Upsampling(img_size, filter=None, factor=2, padding='circular', device='cpu', \*\*kwargs)

Bases: [`Downsampling`](https://deepinv.org/api/stubs/deepinv.physics.Downsampling.html.md#deepinv.physics.Downsampling)

Upsampling operator.

This operator performs the operation

$$
y = h^T * S^T (x)

$$

where $S^T$ is the adjoint of the subsampling operator and $h$ is a low-pass filter.

* **Parameters:**
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – Upsampling filter. It can be `'gaussian'`, `'bilinear'`, `'bicubic'`,
    `'sinc'` or a custom `torch.Tensor` filter. If `None`, no filtering is applied.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the output image
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – upsampling factor
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – options are `'circular'`, `'replicate'` and `'reflect'`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device on which the physics’ buffers will be created. To change the device of the physics, please use `physics.to(device)`.
