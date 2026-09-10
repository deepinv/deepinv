# ADMUNet

### *class* deepinv.models.ADMUNet(img_resolution=64, in_channels=3, out_channels=3, label_dim=0, augment_dim=0, model_channels=192, channel_mult=(1, 2, 3, 4), channel_mult_emb=4, num_blocks=3, attn_resolutions=(32, 16, 8), dropout=0.10, label_dropout=0, pretrained='download', \_was_trained_on_minus_one_one=False, pixel_std=0.75, device=None, \*args, \*\*kwargs)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Implementation of the ADM UNet diffusion model.

From the paper of Dhariwal and Nichol<sup>[1](#footcite-dhariwal2021diffusion)</sup>.

The model is also pre-conditioned by the method described in the EDM paper Karras *et al.*<sup>[2](#footcite-karras2022elucidating)</sup>.

Equivalent to the original implementation by Dhariwal and Nichol, available at: [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion).
The architecture consists of a series of convolution layer, down-sampling residual blocks and up-sampling residual blocks with skip-connections.
Each residual block has a self-attention mechanism with `64` channels per attention head with the up/down-sampling from BigGAN..
The noise level is embedded using Positional Embedding with optional augmentation linear layer.

* **Parameters:**
  * **img_resolution** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Image spatial resolution at input/output.
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of color channels at input.
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of color channels at output.
  * **label_dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of class labels, 0 = unconditional.
  * **augment_dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Augmentation label dimensionality, 0 = no augmentation.
  * **model_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Base multiplier for the number of channels.
  * **channel_mult** (*Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Per-resolution multipliers for the number of channels.
  * **channel_mult_emb** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Multiplier for the dimensionality of the embedding vector.
  * **num_blocks** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of residual blocks per resolution.
  * **attn_resolutions** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – List of resolutions with self-attention.
  * **dropout** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – dropout probability used in residual blocks.
  * **label_dropout** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Dropout probability of class labels for classifier-free guidance.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – use a pretrained network. If `pretrained=None`, the weights will be initialized at random
    using Pytorch’s default initialization. If `pretrained='download'`, the weights will be downloaded from an
    online repository (the default model is a conditional model trained on ImageNet at 64x64 resolution (`imagenet64-cond`) with default architecture).
    Finally, `pretrained` can also be set as a path to the user’s own pretrained weights.
    In this case, the model is supposed to be trained on `[0,1]` pixels, if it was trained on `[-1, 1]` pixels, the user should set the attribute `_was_trained_on_minus_one_one` to `True` after loading the weights.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **\_was_trained_on_minus_one_one** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Indicate whether the model has been trained on `[-1, 1]` pixels or `[0, 1]` pixels. Default to `False`.
  * **pixel_std** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – The standard deviation of the normalized pixels (to `[0, 1]` for example) of the data distribution. Default to `0.75`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Instruct our module to be either on cpu or on gpu. Default to `None`, which suggests working on cpu.

<hr />

* **References:**

* <a id='footcite-dhariwal2021diffusion'>**[1]**</a> Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. *Advances in neural information processing systems*, 34:8780–8794, 2021.
* <a id='footcite-karras2022elucidating'>**[2]**</a> Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. *Advances in neural information processing systems*, 35:26565–26577, 2022.

#### forward(x, sigma, class_labels=None, augment_labels=None, input_in_minus_one_one=False, \*args, \*\*kwargs)

Run the denoiser on noisy image.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image
  * **sigma** (*Union* *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – noise level
  * **class_labels** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – class labels
  * **augment_labels** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – augmentation labels
  * **input_in_minus_one_one** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the input `x` is in `[-1, 1]` range. Default is `False`.
* **Return torch.Tensor:**
  denoised image.

#### forward_unet(x, sigma, class_labels=None, augment_labels=None)

Run the unet on noisy image.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image
  * **sigma** (*Union* *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – noise level
  * **class_labels** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – class labels
  * **augment_labels** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – augmentation labels
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) denoised image.
