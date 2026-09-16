# Restormer

### *class* deepinv.models.Restormer(in_channels=3, out_channels=3, dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4, heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False, pretrained='denoising', device=None)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Restormer denoiser network.

Model introduced by Zamir *et al.*<sup>[1](#footcite-zamir2022restormer)</sup>, specialized in restoration tasks including deraining, single-image motion deblurring,
defocus deblurring and image denoising for high-resolution images.

Code adapted from [https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py](https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py).

By default, the model is a denoising network with pretrained weights. For other tasks such as deraining, some arguments needs to be adapted.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels of the input.
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels of the output.
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels after the first conv operation (`in_channel`, H, W) -> (`dim`, H, W).
    `dim` corresponds to `C` in the figure.
  * **num_blocks** (*Sequence*) – number of `TransformerBlock` for each level of scale in the encoder-decoder stage with a total of 4-level of scales.
    `num_blocks = [L1, L2, L3, L4]` with L1 ≤ L2 ≤ L3 ≤ L4.
  * **num_refinement_blocks** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of `TransformerBlock` in the refinement stage after the decoder stage.
    Corresponds to `Lr` in the figure.
  * **heads** (*Sequence*) – number of heads in `TransformerBlock` for each level of scale in the encoder-decoder stage and in the refinement stage.
    At same scale, all `TransformerBlock` have the same number of heads. The number of heads for the refinement block is `heads[0]`.
  * **ffn_expansion_factor** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – corresponds to $\eta$ in GDFN.
  * **bias** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Add bias or not in each of the Attention and Feedforward layers inside of the `TransformerBlock`.
  * **LayerNorm_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Add bias or not in each of the LayerNorm inside of the `TransformerBlock`.
    `LayerNorm_type = 'BiasFree' / 'WithBias'`.
  * **dual_pixel_task** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Should be true if dual-pixel defocus deblurring is enabled, false for single-pixel deblurring and other tasks.
  * **device** (*None* *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Instruct our module to be either on cpu or on gpu. Default to `None`, which suggests working on cpu.
  * **pretrained** (*None* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Default to `'denoising'`.
    If `pretrained = 'denoising' / 'denoising_gray' / 'denoising_color' / 'denoising_real' / 'deraining' / 'defocus_deblurring'`,
    will download weights from the HuggingFace Hub.
    If `pretrained = '\*.pth'`, will load weights from a local pth file.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – training or testing mode.

#### NOTE
To obtain good performance on a broad range of noise levels, even with limited noise levels during training, it is recommended to remove all additive constants by setting :
`LayerNorm_type='BiasFree'` and `bias=False`, as proposed by Mohan *et al.*<sup>[2](#footcite-mohan2020robust)</sup>.

<hr />

* **References:**

* <a id='footcite-zamir2022restormer'>**[1]**</a> Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: efficient transformer for high-resolution image restoration. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 5728–5739. 2022.
* <a id='footcite-mohan2020robust'>**[2]**</a> Sreyas Mohan, Zahra Kadkhodaie, Eero P Simoncelli, and Carlos Fernandez-Granda. Robust and interpretable blind image denoising via bias-free convolutional neural networks. In *8th International Conference on Learning Representations, ICLR 2020*. 2020.

#### forward(x, sigma=None, \*\*kwargs)

Run the denoiser on noisy image. The noise level is not used in this denoiser.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image

#### forward_restormer(x)

Run the Restormer network on the input image.

The input shape is expected to be divisible by 8.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image

#### is_standard_deblurring_network(in_channels, out_channels, dim, num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type, dual_pixel_task)

Check if model params are the params used to pre-trained the standard network for deblurring.

#### is_standard_denoising_network(in_channels, out_channels, dim, num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type, dual_pixel_task)

Check if model params are the params used to pre-trained the standard network for denoising.

#### is_standard_deraining_network(in_channels, out_channels, dim, num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type, dual_pixel_task)

Check if model params are the params used to pre-trained the standard network for deraining.

<a id="sphx-glr-backref-deepinv-models-restormer"></a>

## Examples using `Restormer`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example focuses on blind image Gaussian denoising, i.e. the problem">  <div class="sphx-glr-thumbnail-title">Blind denoising with noise level estimation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of the denoisers in DeepInverse. A denoiser is a model that takes in a noisy image and outputs a denoised version of it. Basically, it solves the following problem:">  <div class="sphx-glr-thumbnail-title">Benchmarking pretrained denoisers</div>
</div>
<!-- thumbnail-parent-div-close --></div>
