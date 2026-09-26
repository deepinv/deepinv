# IRadon

### *class* deepinv.physics.functional.IRadon(in_size, theta=None, circle=False, use_filter=True, out_size=None, parallel_computation=True, dtype=torch.float, device=torch.device('cpu'))

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Inverse sparse Radon transform operator.

* **Parameters:**
  * **in_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the size of the input image  (assumed square).
  * **theta** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the angles at which the Radon transform is computed. Default is torch.arange(180).
  * **circle** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the input image is assumed to be a circle. Default is False.
  * **use_filter** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the ramp filter is applied to the input image. Default is True.
  * **out_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the size of the output image. If None, the size is the same as the input image.
  * **parallel_computation** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, all projections are performed in parallel. Requires more memory but is faster on GPUs.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – the data type of the output. Default is torch.float.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – the device of the output. Default is torch.device(‘cpu’).

#### forward(x, filtering=True)

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input image.
  * **filtering** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the ramp filter is applied to the input image. Default is True.
