# PhysicsMultiScaler

### *class* deepinv.physics.PhysicsMultiScaler(physics, img_size, filter='sinc', factors=(2, 4, 8), device='cpu', dtype=None, \*\*kwargs)

Bases: [`Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

Multi-scale wrapper for physics operators.

This class applies a physics model at a given scale
by upsampling the input signal before applying the base physics operator.

$$
A(x) = A_{base}(U_{scale}(x))
$$

where $U_{scale}$ is the upsampling operator for the given scale and $A_{base}$ is the base physics operator.

By default, we assume that the factors for the different scales are [2, 4, 8].
The 1st scale corresponds to upsampling by a factor of 2, the 2nd scale corresponds to upsampling by a factor of 4, and so on.
The 0th scale corresponds to the base physics operator without upsampling.

* **Parameters:**
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – base physics operator.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – shape of the input image (C, H, W).
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – type of filter to use for upsampling, e.g., ‘sinc’, ‘nearest’, ‘bilinear’.
  * **factors** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – list of factors to use for upsampling.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device to use for the upsampling operator, e.g., ‘cpu’, ‘mps’, ‘cuda’.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) *,*) – type to be associated with the signal.

#### downsample_measurement(y, scale=None)

Downsample the measurements to a coarser scale

Unlike input images and physics operators, downsampling measurements is
tricky as it depends on the nullspace of the downsampled physics
operator. It is nonetheless possible to compute it for certain physics
operators (blur, inpainting).

By default, this function raises a [`NotImplementedError`](https://docs.python.org/3.9/library/exceptions.html#NotImplementedError) and it
can be reimplemented in subclasses.

#### NOTE
See also specific implementations in
`deepinv.physics.BlurMultiScaler`,
`deepinv.physics.BlurFFTMultiScaler`, and
`deepinv.physics.InpaintingMultiScaler`.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – fine scale measurement
  * **scale** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – target scale in which to express `y`, if None, uses the value of the attribute `scale`, default: None
