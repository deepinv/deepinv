# DownsamplingMatlab

### *class* deepinv.physics.DownsamplingMatlab(factor=2, kernel='cubic', padding='reflect', antialiasing=True, device='cpu', \*\*kwargs)

Bases: [`Downsampling`](https://deepinv.org/api/stubs/deepinv.physics.Downsampling.html.md#deepinv.physics.Downsampling)

Downsampling with MATLAB imresize

Downsamples with default MATLAB `imresize`, using a bicubic kernel, antialiasing and reflect padding.

Wraps `imresize` from a modified version of the [original implementation](https://github.com/sanghyun-son/bicubic_pytorch).

The adjoint is computed using autograd via [`deepinv.physics.adjoint_function()`](https://deepinv.org/api/stubs/deepinv.physics.adjoint_function.html.md#deepinv.physics.adjoint_function).
This is because `imresize` with reciprocal of scale is not a correct adjoint.
Note however the adjoint is quite slow.

* **Parameters:**
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – downsampling factor
  * **kernel** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – MATLAB kernel, supports only `cubic` for bicubic downsampling.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – MATLAB padding type, supports only `reflect` for reflect padding.
  * **antialiasing** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to perform antialiasing in MATLAB downsampling.
    Recommended to set to `True` to match MATLAB.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device on which the physics’ buffers will be created. If a buffer is updated via `physics.update_parameters()`, if not None, it will be automatically casted to the device of the replaced buffer, else, use the device of the provided value. To change the device of all buffers, please use `physics.to(device)`.

#### A(x, factor=None, \*\*kwargs)

Downsample forward operator

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – downsampling factor. If not `None`, use this factor and store it as current factor.

#### A_adjoint(y, factor=None, \*\*kwargs)

Downsample adjoint operator via autograd.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurement
  * **factor** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – downsampling factor. If not `None`, use this factor and store it as current factor.

<a id="sphx-glr-backref-deepinv-physics-downsamplingmatlab"></a>

## Examples using `DownsamplingMatlab`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div>
<!-- thumbnail-parent-div-close --></div>
