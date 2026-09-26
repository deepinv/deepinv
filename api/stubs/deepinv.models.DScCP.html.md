# DScCP

### *class* deepinv.models.DScCP(depth=20, n_channels_per_layer=64, pretrained='download', pretrained_2d_isotropic=False, device=None, dim=2)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

DScCP denoiser network.

The network architecture is based on the paper from Le *et al.*<sup>[1](#footcite-le2024unfolded)</sup>.
and has an unrolled architecture based on the fast Chambolle-Pock algorithm using strong convexity.
DScCP stands for Deep Strongly Convex Chambolle Pock.

The pretrained weights are trained with the default parameters of the network, i.e. depth=20 layers, n_channels_per_layer=64 channels.
They can be downloaded via setting `pretrained='download'`.

* **Parameters:**
  * **depth** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – depth i.e. number of convolutional layers.
  * **n_channels_per_layer** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels per convolutional layer.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – `pretrained='download'` to download pretrained weights, or path to local weights file. When building a 3D network, it is possible to initialize with 2D pretrained weights by using `pretrained='download_2d'`, which provides a good starting point for fine-tuning.
  * **pretrained_2d_isotropic** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – when loading 2D pretrained weights into a 3D network, whether to initialize the 3D kernels isotropically. By default the weights are loaded axially, i.e., by initializing the central slice of the 3D kernels with the 2D weights.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – ‘cuda’, ‘mps’ or ‘cpu’.
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **References:**

* <a id='footcite-le2024unfolded'>**[1]**</a> Hoang Trieu Vy Le, Audrey Repetti, and Nelly Pustelnik. Unfolded proximal neural networks for robust image gaussian denoising. *IEEE Transactions on Image Processing*, 2024.

#### forward(x, sigma=0.03)

Run the denoiser on noisy image.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level.
