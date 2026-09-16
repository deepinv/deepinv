# KernelIdentificationNetwork

### *class* deepinv.models.KernelIdentificationNetwork(filters=25, blur_kernel_size=33, bilinear=False, no_softmax=False, pretrained='download', device='cpu')

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Space varying blur kernel estimation network.

U-Net proposed by Carbajal *et al.*<sup>[1](#footcite-carbajal2023blind)</sup>, estimating
the parameters of [`deepinv.physics.SpaceVaryingBlur`](https://deepinv.org/api/stubs/deepinv.physics.SpaceVaryingBlur.html.md#deepinv.physics.SpaceVaryingBlur) forward model, i.e., blur kernels and corresponding spatial multipliers (masks).

#### NOTE
The estimated parameters should therefore be plugged into
`deepinv.physics.SpaceVaryingBlur(mask_first=False, padding="circular")`.

Current implementation supports blur kernels of size 33x33 (default) and 65x65, and 1 or 3 input channels.

Code adapted from [https://github.com/GuillermoCarbajal/J-MKPD](https://github.com/GuillermoCarbajal/J-MKPD) with permission from the author.

Images are assumed to be in range [0, 1] before being passed to the network, and to be **non-gamma corrected** (i.e., linear RGB).
If your blurry image has been gamma-corrected (e.g., standard sRGB images), consider applying an inverse gamma correction (e.g., raising to the power of 2.2)
before passing it to the network for better results.

* **Parameters:**
  * **filters** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of blur kernels to estimate, defaults to 25.
  * **blur_kernel_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – size of the blur kernels to estimate, defaults to 33. Only 33 and 65 are currently supported.
  * **bilinear** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use bilinear upsampling or transposed convolutions, defaults to False.
  * **no_softmax** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to apply softmax to the estimated kernels, defaults to False.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – use a pretrained network. If `pretrained=None`, the weights will be initialized at random
    using Pytorch’s default initialization. If `pretrained='download'`, the weights will be downloaded from an
    online repository (only available for the default architecture with default parameters).
    Finally, `pretrained` can also be set as a path to the user’s own pretrained weights.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device to use, defaults to ‘cpu’.

<hr />

* **Examples:**
  ```pycon
  >>> import deepinv as dinv
  >>> import torch
  >>> device = "cuda" if torch.cuda.is_available() else "cpu"
  >>> kernel_estimator = dinv.models.KernelIdentificationNetwork(device=device)
  >>> physics = dinv.physics.SpaceVaryingBlur(device=device, padding="circular", mask_first=False)
  >>> y = torch.randn(1, 3, 128, 128).to(device)  # random blurry image for demonstration
  >>> with torch.no_grad():
  ...     params = kernel_estimator(y)  # this outputs {"filters": ..., "multipliers": ...}
  >>> physics.update(**params) # update physics with estimated kernels
  >>> print(params["filters"].shape, params["multipliers"].shape)
  torch.Size([1, 1, 25, 33, 33]) torch.Size([1, 1, 25, 128, 128])
  ```

<hr />

* **References:**

* <a id='footcite-carbajal2023blind'>**[1]**</a> Guillermo Carbajal, Patricia Vitoria, José Lezama, and Pablo Musé. Blind motion deblurring with pixel-wise kernel estimation via kernel prediction networks. *IEEE Transactions on Computational Imaging*, 9:928–943, 2023.

#### forward(x)

Forward pass of the kernel estimation network.

* **Parameters:**
  **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input blurry image of shape (N, C, H, W) with values in [0, 1]. Assumed to be non-gamma corrected (i.e., linear RGB).
* **Returns:**
  dictionary with estimated blur kernels and spatial multipliers of the
  $y = \sum_k w_k \odot (h_k \star x)$ model:
  -  `'filters'`: estimated blur kernels $h_k$ of shape (N, 1, K, blur_kernel_size, blur_kernel_size)
  -  `'multipliers'`: estimated spatial masks $w_k$ of shape (N, 1, K, H, W)
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)[[str](https://docs.python.org/3.9/library/stdtypes.html#str), [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

<a id="sphx-glr-backref-deepinv-models-kernelidentificationnetwork"></a>

## Examples using `KernelIdentificationNetwork`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div>
<!-- thumbnail-parent-div-close --></div>
