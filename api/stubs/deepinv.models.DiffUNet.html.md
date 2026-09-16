# DiffUNet

### *class* deepinv.models.DiffUNet(in_channels=3, out_channels=3, large_model=False, use_fp16=False, pretrained='download')

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Diffusion UNet model.

This is the model with attention and timestep embeddings from Choi *et al.*<sup>[1](#footcite-choi2021ilvr)</sup>;
code is adapted from [https://github.com/jychoi118/ilvr_adm](https://github.com/jychoi118/ilvr_adm).

It is possible to choose the standard model from Choi *et al.*<sup>[1](#footcite-choi2021ilvr)</sup> with 128 hidden channels per layer (trained on FFHQ)
and a larger model Dhariwal and Nichol<sup>[2](#footcite-dhariwal2021diffusion)</sup> with 256 hidden channels per layer (trained on ImageNet128).

A pretrained network for (in_channels=out_channels=3)
can be downloaded via setting `pretrained='download'`.

The network can handle images of size $2^{n_1}\times 2^{n_2}$ with $n_1,n_2 \geq 5$.

#### NOTE
The weights available for download are pretrained on 256x256 images,
thus generation is likely to fail for different image sizes
(see [https://github.com/deepinv/deepinv/issues/602](https://github.com/deepinv/deepinv/issues/602)).

#### WARNING
This model has 2 forward modes:

* `forward_diffusion`: in the first mode, the model takes a noisy image and a timestep as input and estimates the noise map in the input image. This mode is consistent with the original implementation from the authors, i.e. it assumes the same image normalization.
* `forward_denoise`: in the second mode, the model takes a noisy image and a noise level as input and estimates the noiseless underlying image in the input image. In this case, we assume that images have values in [0, 1] and a rescaling is performed under the hood.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – channels in the input Tensor.
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – channels in the output Tensor.
  * **large_model** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, use the large model with 256 hidden channels per layer trained on ImageNet128
    (weights size: 2.1 GB).
    Otherwise, use a smaller model with 128 hidden channels per layer trained on FFHQ (weights size: 357 MB).
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – use a pretrained network. If `pretrained=None`, the weights will be initialized at
    random using Pytorch’s default initialization.
    If `pretrained='download'`, the weights will be downloaded from an online repository
    (only available for 3 input and output channels).
    Finally, `pretrained` can also be set as a path to the user’s own pretrained weights.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.

<hr />

* **References:**

* <a id='footcite-choi2021ilvr'>**[1]**</a> Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: conditioning method for denoising diffusion probabilistic models. In *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, 14347–14356. IEEE, 2021.
* <a id='footcite-dhariwal2021diffusion'>**[2]**</a> Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. *Advances in neural information processing systems*, 34:8780–8794, 2021.

#### convert_to_fp16()

Convert the tensor of the model to float16.

#### convert_to_fp32()

Convert the tensor of the model to float32.

#### find_nearest(array, value)

Find the argmin of the nearest value in a tensor.

#### forward(x, t, y=None, type_t='noise_level')

Apply the model to an input batch.

This function takes a noisy image and either a timestep or a noise level as input. Depending on the nature of
`t`, the model returns either a noise map (if `type_t='timestep'`) or a denoised image (if
`type_t='noise_level'`).

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an `(N, C, ...)` Tensor of inputs.
  * **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – a 1-D batch of timesteps or noise levels.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an (N) Tensor of labels, if class-conditional. Default=None.
  * **type_t** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Nature of the embedding `t`. In traditional diffusion model, and in the authors’ code, `t` is
    a timestep linked to a noise level; in this case, set `type_t='timestep'`. We can also choose
    `t` to be a noise level directly and use the model as a denoiser; in this case, set
    `type_t='noise_level'`. Default: `'timestep'`.
* **Returns:**
  an `(N, C, ...)` Tensor of outputs. Either a noise map (if `type_t='timestep'`) or a denoised image
  (if `type_t='noise_level'`).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward_denoise(x, sigma, y=None)

Applies the denoising model to an input batch.

This function takes a noisy image and a noise level as input (and not a timestep) and estimates the noiseless
underlying image in the input image.
The input image is assumed to be in range [0, 1] (up to noise) and to have dimensions with width and height
divisible by a power of 2.

#### NOTE
The DiffUNet assumes that images are scaled as $\sqrt{\alpha_t} x + (1-\alpha_t) \epsilon$
thus an additional rescaling by $\sqrt{\alpha_t}$ is performed within this function, along with
a mean shift by correction by $0.5 - \sqrt{\alpha_t} 0.5$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an `(N, C, ...)` Tensor of inputs.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – a 1-D batch of noise levels.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an (N) Tensor of labels, if class-conditional. Default=None.
* **Returns:**
  an `(N, C, ...)` Tensor of outputs.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### forward_diffusion(x, timesteps, y=None)

Apply the model to an input batch.

This function takes a noisy image and a timestep as input (and not a noise level) and estimates the noise map
in the input image.
The image is assumed to be in range [-1, 1] and to have dimensions with width and height divisible by a
power of 2.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an [N x C x …] Tensor of inputs.
  * **timesteps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – a 1-D batch of timesteps.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an [N] Tensor of labels, if class-conditional. Default=None.
* **Returns:**
  an `(N, 2*C, ...)` Tensor of outputs, where the first C
  channels are the noise estimates and the remaining C are the per-pixel
  variances, as in the original implementation:
  [https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py#L263](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py#L263)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### get_alpha_prod(beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=1000)

Get the alpha sequences; this is necessary for mapping noise levels to timesteps when performing pure denoising.

#### patch_forward(x, t, y=None, type_t='noise_level', patch_size=512)

Splits an image tensor into patches (without overlapping), applies the model to each patch, and reconstructs the full image.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input low-quality image tensor of shape (B, C, H, W).
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Size of the patches to split into.
  * **\*args** – Additional positional arguments for the model.
  * **\*\*kwargs** – Additional keyword arguments for the model.
* **Returns:**
  Reconstructed image tensor.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-diffunet"></a>

## Examples using `DiffUNet`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from :footcitezhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div>
<!-- thumbnail-parent-div-close --></div>
