# NCSNpp

### *class* deepinv.models.NCSNpp(model_type='ncsn', precondition_type='edm', img_resolution=64, in_channels=3, out_channels=3, label_dim=0, augment_dim=9, model_channels=128, channel_mult=(1, 2, 2, 2), channel_mult_emb=4, num_blocks=4, attn_resolutions=(16,), dropout=0.10, label_dropout=0.0, pretrained='download', \_was_trained_on_minus_one_one=False, pixel_std=0.75, device=None, \*\*kwargs)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Implementation of the DDPM++ and NCSN++ UNet architectures.

Equivalent to the original implementation by Song *et al.*<sup>[1](#footcite-song2020score)</sup>, available at [the official implementation](https://github.com/yang-song/score_sde_pytorch).
The DDPM model was originally built for the VP-SDE from Song *et al.*<sup>[1](#footcite-song2020score)</sup> while the NCSN++ model was originally built with the VE-SDE.
See the [diffusion SDE implementations](https://deepinv.org/user_guide/reconstruction/sampling.html.md#diffusion) for more details on the VP-SDE and VE-SDE from Song *et al.*<sup>[1](#footcite-song2020score)</sup>.
The model is also pre-conditioned by the method described in Karras *et al.*<sup>[2](#footcite-karras2022elucidating)</sup>.

The architecture consists of a series of convolution layer, down-sampling residual blocks and up-sampling residual blocks with skip-connections of scale $\sqrt{0.5}$.
The model also supports an additional class condition model.
Each residual block has a self-attention mechanism with multiple channels per attention head.
The noise level can be embedded using either Positional Embedding  or Fourier Embedding with optional augmentation linear layer.

* **Parameters:**
  * **model_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – 

    Model type, which defines the architecture and embedding types. Options are:
    - `'ncsn'` for the NCSN++ architecture: the following arguments will be ignored and set to `embedding_type='fourier'`, `channel_mult_noise=2`, `encoder_type='residual'`, `decoder_type='standard'`, `resample_filter=[1,3,3,1]`.
    - `'ddpm'` for the  DDPM++ architecture: the following arguments will be ignored and set to `embedding_type='positional'`, `channel_mult_noise=1`, `encoder_type='standard'`, `decoder_type='standard'`, `resample_filter=[1,1]`.

    Default is `'ncsn'`.
  * **precondition_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Input preconditioning for denoising. Can be ‘edm’ for the method from Karras *et al.*<sup>[2](#footcite-karras2022elucidating)</sup> or ‘baseline_ve’ for the original method from Song *et al.*<sup>[1](#footcite-song2020score)</sup>. See Table 1 from Karras *et al.*<sup>[2](#footcite-karras2022elucidating)</sup> for more details.
  * **img_resolution** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Image spatial resolution at input/output.
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of color channels at input.
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of color channels at output.
  * **label_dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of class labels, 0 = unconditional.
  * **augment_dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Augmentation label dimensionality, 0 = no augmentation.
  * **model_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Base multiplier for the number of channels.
  * **channel_mult** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – Per-resolution multipliers for the number of channels.
  * **channel_mult_emb** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Multiplier for the dimensionality of the embedding vector.
  * **num_blocks** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of residual blocks per resolution.
  * **attn_resolutions** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – List of resolutions with self-attention.
  * **dropout** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Dropout probability of intermediate activations.
  * **label_dropout** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Dropout probability of class labels for classifier-free guidance.
  * **embedding_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Timestep embedding type: `'positional'` for DDPM++, `'fourier'` for NCSN++.
  * **channel_mult_noise** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
  * **encoder_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Encoder architecture: `'standard'` for DDPM++, `'residual'` for NCSN++.
  * **decoder_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Decoder architecture: `'standard'` for both DDPM++ and NCSN++.
  * **resample_filter** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – Resampling filter: `[1,1]` for DDPM++, `[1,3,3,1]` for NCSN++.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – 

    Use pretrained weights (or a path to custom weights).
    - If `pretrained is None`, the weights are initialized randomly using PyTorch’s default initialization.
    - `pretrained='edm-ffhq64-64x64-uncond-ve'` loads **NCSN++** weights from Karras *et al.*<sup>[2](#footcite-karras2022elucidating)</sup>, trained on **FFHQ 64x64** with the **EDM** diffusion schedule (see Table 1 in Karras *et al.*<sup>[2](#footcite-karras2022elucidating)</sup>).
    - `pretrained='edm-cifar10-32x32-uncond-ve'` loads **NCSN++** weights from Karras *et al.*<sup>[2](#footcite-karras2022elucidating)</sup>, trained on **CIFAR-10 32x32** with the **EDM** diffusion schedule.
    - `pretrained='edm-ffhq-64x64-uncond-vp'` loads **DDPM++** weights from Karras *et al.*<sup>[2](#footcite-karras2022elucidating)</sup>, trained on **FFHQ 64x64** with the **EDM** diffusion schedule.
    - `pretrained='edm-cifar10-32x32-uncond-vp'` loads **DDPM++** weights from Karras *et al.*<sup>[2](#footcite-karras2022elucidating)</sup>, trained on **CIFAR-10 32x32** with the **EDM** diffusion schedule.
    - `pretrained='baseline-ffhq-64x64-uncond-ve'` loads **NCSN++** weights from Song *et al.*<sup>[1](#footcite-song2020score)</sup>, trained on **FFHQ 64x64** with the **VE-SDE** diffusion schedule.
    - `pretrained='baseline-cifar10-32x32-uncond-ve'` loads **NCSN++** weights from Song *et al.*<sup>[1](#footcite-song2020score)</sup>, trained on **CIFAR-10 32x32** with the **VE-SDE** diffusion schedule.
    - `pretrained='download'` is a convenience alias: if `model_type='ncsn'` (default) it maps to `'edm-ffhq-64x64-uncond-ve'`, and if `model_type='ddpm'` it maps to `'edm-ffhq-64x64-uncond-vp'`.
    - `pretrained` may also be a filesystem path to user-provided weights; the model is assumed to be trained on pixels in `[0, 1]`—if trained on `[-1, 1]`, set `model._was_trained_on_minus_one_one = True` after loading.

    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **\_was_trained_on_minus_one_one** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Indicate whether the model has been trained on `[-1, 1]` pixels or `[0, 1]` pixels. Default to `False`.
  * **pixel_std** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – The standard deviation of the normalized pixels (to `[0, 1]` for example) of the data distribution. Default to `0.75`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Instruct our module to be either on cpu or on gpu. Default to `None`, which suggests working on cpu.

<hr />

* **References:**

* <a id='footcite-song2020score'>**[1]**</a> Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*. 2020.
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

Run the unet.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image
  * **sigma** (*Union* *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *]*) – noise level
  * **class_labels** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – class labels
  * **augment_labels** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – augmentation labels
* **Return torch.Tensor:**
  denoised image.

<a id="sphx-glr-backref-deepinv-models-ncsnpp"></a>

## Examples using `NCSNpp`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div>
<!-- thumbnail-parent-div-close --></div>
