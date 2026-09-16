# DRUNet

### *class* deepinv.models.DRUNet(in_channels=3, out_channels=3, nc=(64, 128, 256, 512), nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', pretrained='download', pretrained_2d_isotropic=False, device=None, dim=2)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

DRUNet denoiser network.

The network architecture is based on the paper Zhang *et al.*<sup>[1](#footcite-zhang2021plug)</sup>.
and has a U-Net like structure, with convolutional blocks in the encoder and decoder parts.

The network takes into account the noise level of the input image, which is encoded as an additional input channel.

A pretrained network for (in_channels=out_channels=1 or in_channels=out_channels=3)
can be downloaded via setting `pretrained='download'`.

#### TIP
This model can handle non-uniform `sigma` maps, which can be of size `(batch_size, 1, height, width)`.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels of the input.
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels of the output.
  * **nc** (*Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – number of channels per convolutional layer, the network has a fixed number of 4 scales with `nb` blocks per scale (default: `[64,128,256,512]`).
  * **nb** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of convolutional blocks per layer.
  * **act_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – activation mode, “R” for ReLU, “L” for LeakyReLU “E” for ELU and “s” for Softplus.
  * **downsample_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Downsampling mode, “avgpool” for average pooling, “maxpool” for max pooling, and
    “strideconv” for convolution with stride 2.
  * **upsample_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Upsampling mode, “convtranspose” for convolution transpose, “pixelshuffle” for pixel
    shuffling, and “upconv” for nearest neighbour upsampling with additional convolution. “pixelshuffle” is not implemented for 3D.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – use a pretrained network. If `pretrained=None`, the weights will be initialized at random
    using Pytorch’s default initialization. If `pretrained='download'`, the weights will be downloaded from an
    online repository (only available for the default architecture with 3 or 1 input/output channels). When building a 3D network, it is possible to initialize with 2D pretrained weights by using `pretrained='download_2d'`, which provides a good starting point for fine-tuning.
    Finally, `pretrained` can also be set as a path to the user’s own pretrained weights.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **pretrained_2d_isotropic** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – when loading 2D pretrained weights into a 3D network, whether to initialize the 3D kernels isotropically. By default the weights are loaded axially, i.e., by initializing the central slice of the 3D kernels with the 2D weights.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to put the model on.
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **Examples:**
  ```pycon
  >>> import deepinv as dinv
  >>> import torch
  >>> denoiser = dinv.models.DRUNet()
  >>> y = torch.randn(1, 3, 32, 32)
  >>> sigma = 0.1
  >>> with torch.no_grad():
  ...     denoised = denoiser(y, sigma)
  ```

<hr />

* **References:**

* <a id='footcite-zhang2021plug'>**[1]**</a> Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte. Plug-and-play image restoration with deep denoiser prior. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(10):6360–6376, 2021.

#### forward(x, sigma)

Run the denoiser on image with noise level $\sigma$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noise level. If `sigma` is a float, it is used for all images in the batch.
    If `sigma` is a tensor, it can be of shape `(batch_size,)` or `(batch_size, 1, height, width, (depth))`.

<a id="sphx-glr-backref-deepinv-models-drunet"></a>

## Examples using `DRUNet`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example focuses on blind image Gaussian denoising, i.e. the problem">  <div class="sphx-glr-thumbnail-title">Blind denoising with noise level estimation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many imaging problems, the data to be processed can be very large, making it challenging to fit the denoising process into the memory of a single device. For instance, medical imaging or satellite imagery often involves processing gigapixel images that cannot be processed as a whole.">  <div class="sphx-glr-thumbnail-title">Distributed Denoiser with Image Tiling</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to fit deepinv.loss.metric.NIQE on a new dataset, and use it to evaluate denoiser performance.">  <div class="sphx-glr-thumbnail-title">Fitting NIQE on a custom dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of the denoisers in DeepInverse. A denoiser is a model that takes in a noisy image and outputs a denoised version of it. Basically, it solves the following problem:">  <div class="sphx-glr-thumbnail-title">Benchmarking pretrained denoisers</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise &lt;deepinv.physics.PoissonGaussianNoise&gt; using the Generalized Anscombe Transform (GAT) &lt;deepinv.models.AnscombeDenoiser&gt;, which converts any Gaussian denoiser into a Poisson-Gaussian denoiser :footcitemakitalo2012optimal.">  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm :footcitekawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div>
<!-- thumbnail-parent-div-close --></div>
