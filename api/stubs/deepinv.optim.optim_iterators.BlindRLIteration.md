# BlindRLIteration

### *class* deepinv.optim.optim_iterators.BlindRLIteration(k_prior=None, normalize_kernel=True, use_fft=False, eps=1e-8, \*\*kwargs)

Bases: [`OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.md#deepinv.optim.OptimIterator)

Iterator for Blind Richardson-Lucy deconvolution.

This iterator performs one step to estimate the next kernel, and one step to
estimate the next image.

The current iterate is stored as `X["est"] = (x, k)`. The kernel update
assumes 2D circular convolution and a spatially invariant kernel shared by
all image channels.

* **Parameters:**
  * **k_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.md#deepinv.optim.Prior) *,* *None*) – optional kernel prior. Default: `None`.
  * **normalize_kernel** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to normalize the kernel to unit sum. Default: `True`.
  * **use_fft** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use the FFT implementations for convolutions. Default: `False`.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – numerical stability constant used for divisions. Default: `1e-8`.

#### forward(X, cur_data_fidelity, cur_prior, cur_params, y, physics, \*args, \*\*kwargs)

Single Blind Richardson-Lucy iteration.

* **Parameters:**
  * **X** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*  *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – Current
    iterate with `X["est"] = (x, k)`.
  * **cur_data_fidelity** ([*deepinv.optim.DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.md#deepinv.optim.DataFidelity)) – Data fidelity term.
  * **cur_prior** ([*deepinv.optim.Prior*](https://deepinv.org/api/stubs/deepinv.optim.Prior.md#deepinv.optim.Prior)) – Image prior.
  * **cur_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Parameters containing `x_steps`, `k_steps`,
    `lambda_reg_x`, `lambda_reg_k`, `g_param` and
    `g_param_kernel`.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Blurry observation of shape `(B, C, H, W)`.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics)) – Blur physics updated in-place with the
    current kernel for the image update.
* **Returns:**
  Dictionary `{"est": (x, k), "cost": F, "it": it}` containing
  the updated image, kernel, cost, and iteration number.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [tuple](https://docs.python.org/3.9/library/stdtypes.html#tuple)[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]
