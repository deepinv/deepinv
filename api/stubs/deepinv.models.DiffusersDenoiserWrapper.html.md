# DiffusersDenoiserWrapper

### *class* deepinv.models.DiffusersDenoiserWrapper(mode_id=None, clip_output=True, dtype=torch.float32, device='cpu', \*args, \*\*kwargs)

Bases: [`ScoreModelWrapper`](https://deepinv.org/api/stubs/deepinv.models.ScoreModelWrapper.html.md#deepinv.models.ScoreModelWrapper)

Wraps a [HuggingFace diffusers](https://huggingface.co/docs/diffusers/index) model as a DeepInv Denoiser.

* **Parameters:**
  * **mode_id** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Diffusers model id or HuggingFace hub repository id. For example, ‘google/ddpm-cat-256’.
    The id must work with `DiffusionPipeline`.
    See [Diffusers Documentation](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/overview#diffusers.DiffusionPipeline).
  * **clip_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to clip the output to the model range. Default is `True`.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device to load the model on. Default is ‘cpu’.

#### NOTE
Currently, only models trained with `DDPMScheduler`, `DDIMScheduler` or `PNDMScheduler` are supported.

#### WARNING
This wrapper requires the `diffusers` and `transformers` packages.
You can install them via `pip install diffusers transformers`.

<hr />

* **Examples:**
  ```pycon
  >>> import deepinv as dinv
  >>> from deepinv.models import DiffusersDenoiserWrapper
  >>> import torch
  >>> device = dinv.utils.get_device(verbose=False)
  >>> denoiser = DiffusersDenoiserWrapper(mode_id='google/ddpm-cat-256', device=device)
  >>> x = dinv.utils.load_example(
  ...         "cat.jpg",
  ...         img_size=256,
  ...         resize_mode="resize",
  ...     ).to(device)
  ```

  ```pycon
  >>> sigma = 0.1
  >>> x_noisy = x + sigma * torch.randn_like(x)
  >>> with torch.no_grad():
  ...     x_denoised = denoiser(x_noisy, sigma=sigma)
  ```

#### forward(x, sigma=None, \*args, \*\*kwargs)

Applies denoiser $\denoiser{x}{\sigma}$.
The input `x` is expected to be in `[0, 1]` range (up to random noise) and the output is also in `[0, 1]` range.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy input, of shape `[B, C, H, W]`.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level. Can be a `float` or a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `[B]`.
    If a single `float` is provided, the same noise level is used for all samples in the batch.
    Otherwise, batch-wise noise levels are used.
  * **args** – additional positional arguments to be passed to the model.
  * **kwarg** – additional keyword arguments to be passed to the model. For example, a `prompt` for text-conditioned or `class_label` for class-conditioned models.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the denoised output.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-diffusersdenoiserwrapper"></a>

## Examples using `DiffusersDenoiserWrapper`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
