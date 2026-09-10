# SwinIR

### *class* deepinv.models.SwinIR(img_size=128, patch_size=1, in_chans=3, embed_dim=180, depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6), window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1.0, upsampler='', resi_connection='1conv', pretrained='download', pretrained_noise_level=15, \*\*kwargs)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

SwinIR denoising network.

The Swin Image Restoration (SwinIR) denoising network was introduced by Liang *et al.*<sup>[1](#footcite-liang2021swinir)</sup>. This code is adapted from the official implementation by the
authors.

* **Parameters:**
  * **img_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Input image size. Default 128.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Patch size. Default: 1.
  * **in_chans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of input image channels. Default: 3.
  * **embed_dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Patch embedding dimension. Default: 180.
  * **depths** (*Sequence*) – Depth of each Swin Transformer layer.
  * **num_heads** (*Sequence*) – Number of attention heads in different layers.
  * **window_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Window size. Default: 8.
  * **mlp_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Ratio of mlp hidden dim to embedding dim. Default: 2.
  * **qkv_bias** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, add a learnable bias to query, key, value. Default: True.
  * **qk_scale** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Override default qk scale of head_dim \*\* -0.5 if set. Default: None.
  * **drop_rate** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Dropout rate. Default: 0.
  * **attn_drop_rate** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Attention dropout rate. Default: 0.
  * **drop_path_rate** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Stochastic depth rate. Default: 0.1.
  * **norm_layer** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Normalization layer. Default: nn.LayerNorm.
  * **ape** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, add absolute position embedding to the patch embedding. Default: False.
  * **patch_norm** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, add normalization after patch embedding. Default: True.
  * **use_checkpoint** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to use checkpointing to save memory. Default: False.
  * **upscale** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
  * **img_range** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Image range. 1. or 255. Default: 1.
  * **upsampler** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – The reconstruction module. ‘’/’pixelshuffle’/’pixelshuffledirect’/’nearest+conv’/None.
    Default: ‘’.
  * **resi_connection** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – The convolutional block before residual connection. Should be either ‘1conv’ or ‘3conv’.
    Default: ‘1conv’.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – Use a pretrained network. If `pretrained=None`, the weights will be initialized at
    random using PyTorch’s default initialization. If `pretrained='download'`, the weights will be downloaded from
    the authors’ online repository [https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) (only available for the
    default architecture). Finally, `pretrained` can also be set as a path to the user’s own pretrained weights.
    Default: ‘download’.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **pretrained_noise_level** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – The noise level of the pretrained model to be downloaded (in 0-255 scale). This
    value is directly concatenated to the download url; should be chosen in the set {15, 25, 50}. Default: 15.

#### NOTE
This class requires the `timm` package to be installed. Install with `pip install timm`.

<hr />

* **References:**

* <a id='footcite-liang2021swinir'>**[1]**</a> Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: image restoration using swin transformer. In *Proceedings of the IEEE/CVF international conference on computer vision*, 1833–1844. 2021.

#### forward(x, sigma=None, \*\*kwargs)

Run the denoiser on noisy image. The noise level is not used in this denoiser.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image, of shape B, C, W, H.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level (not used).

<a id="sphx-glr-backref-deepinv-models-swinir"></a>

## Examples using `SwinIR`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of the denoisers in DeepInverse. A denoiser is a model that takes in a noisy image and outputs a denoised version of it. Basically, it solves the following problem:">  <div class="sphx-glr-thumbnail-title">Benchmarking pretrained denoisers</div>
</div>
<!-- thumbnail-parent-div-close --></div>
