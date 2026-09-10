# MinusOneOneDenoiserWrapper

### *class* deepinv.models.MinusOneOneDenoiserWrapper(model, xmin=0.0, xmax=1.0)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

A wrapper for denoisers trained on $[x_{\mathrm{min}}, x_{\mathrm{max}}]$ images to be used with math:`[-1, 1]` images, i.e. on diffusion sampling iterates.

* **Parameters:**
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – the denoiser to be wrapped.
  * **xmin** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – minimum value of the denoiser training range. Default to `0.0`.
  * **xmax** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum value of the denoiser training range. Default to `1.0`.
