# MinusOneOneDenoiserWrapper

### *class* deepinv.models.MinusOneOneDenoiserWrapper(model, xmin=0.0, xmax=1.0)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

A wrapper for denoisers trained on $[x_{\mathrm{min}}, x_{\mathrm{max}}]$ images to be used with math:`[-1, 1]` images, i.e. on diffusion sampling iterates.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – the denoiser to be wrapped.
  * **xmin** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – minimum value of the denoiser training range. Default to `0.0`.
  * **xmax** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum value of the denoiser training range. Default to `1.0`.

#### forward(x, sigma, \*args, \*\*kwargs)

Apply the wrapped denoiser to the input.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input image.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the noise level.
  * **args** – additional arguments to pass to the denoiser.
  * **kwargs** – additional keyword arguments passed to the denoiser. If `input_in_minus_one_one=True`, the input is assumed to be in `[-1, 1]`; otherwise it is assumed to be in `[0, 1]` and converted to the denoiser’s training range.
* **Returns:**
  the denoised image.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
