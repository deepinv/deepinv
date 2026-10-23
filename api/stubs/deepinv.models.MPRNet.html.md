# MPRNet

### *class* deepinv.models.MPRNet(in_channels=3, out_channels=3, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False, pretrained=None, device='cpu', dim=2)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor), [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

MPRNet restoration model.

MPRNet is a blind restoration model that was proposed in Zamir *et al.*<sup>[1](#footcite-zamir2021multi)</sup>.

There are pretrained weights available for two tasks, which can be loaded via `pretrained`:

> - Real-world image denoising on SIDD: `pretrained='denoising'` with the default architecture.
> - Image deblurring: `pretrained='deblurring'` with `n_feat=96` and default values for the other arguments.

#### NOTE
All spatial dimensions of the input (height and width, and depth for 3D
networks) must be divisible by 8, otherwise an error is raised.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of channels in the input image.
  * **out_channels** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of channels in the output image.
  * **n_feat** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of feature maps in the intermediate layers.
  * **scale_unetfeats** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of feature maps added at each level of the U-Net
    encoder and decoder stages.
  * **scale_orsnetfeats** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of feature maps added to the original resolution
    stage.
  * **num_cab** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – number of channel attention blocks in each original resolution
    block.
  * **kernel_size** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – kernel size of the convolutional layers.
  * **reduction** ([*int*](https://docs.python.org/3.12/library/functions.html#int)) – reduction factor of the channel attention layers.
  * **bias** ([*bool*](https://docs.python.org/3.12/library/functions.html#bool)) – use learnable biases in the convolutional layers.
  * **pretrained** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* *None*) – use a pretrained network. If `pretrained=None`, the weights will be initialized at random
    using Pytorch’s default initialization. If `pretrained='denoising'` or `pretrained='deblurring'`, the weights
    will be downloaded from an online repository, which requires the architecture of the original paper (for
    `'deblurring'`, set `n_feat=96`). Finally, `pretrained` can also be set as a path to the user’s own
    pretrained weights. See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.12/library/stdtypes.html#str)) – device to put the model on.
  * **dim** ([*str*](https://docs.python.org/3.12/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.12/library/functions.html#int)) – whether to build a 2D or 3D network (if str, can be “2”, “2d”,
    “3D”, etc.).

<hr />

* **References:**

* <a id='footcite-zamir2021multi'>**[1]**</a> Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Multi-stage progressive image restoration. In *2021 IEEE/CVF conference on computer vision and pattern recognition (CVPR)*, 14816–14826. IEEE, 2021.

#### forward(x, physics=None, \*\*kwargs)

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Image to be restored.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward operator (not used).
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) restored image estimate.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
