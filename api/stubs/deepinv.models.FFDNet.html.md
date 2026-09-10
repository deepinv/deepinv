# FFDNet

### *class* deepinv.models.FFDNet(n_conv_layers=15, nf=64, img_channels=1, residual_denoising=False, norm=None, orthogonal_init=True, last_conv_bias=True, pretrained='download', device='cpu')

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

FFDNet denoiser network.

The network architecture is based on the paper Zhang *et al.*<sup>[1](#footcite-zhang2018ffdnet)</sup>.
and consists of a `PixelUnshuffle` downsampling operation, a series of 3x3 convolutional layers
(similar to DnCNN), followed by a `PixelShuffle` upsampling operation to get back to the original shape.

The network takes into account the noise level of the input image, which is encoded as an additional input channel.

By default, pretrained grayscale weights are downloaded (`pretrained='download'`). Pretrained weights are
also available for RGB images:

- **grayscale** (default): `FFDNet(n_conv_layers=15, nf=64, img_channels=1, norm=None, last_conv_bias=True, pretrained='download')`
- **color**: `FFDNet(n_conv_layers=12, nf=96, img_channels=3, norm=None, last_conv_bias=True, pretrained='download')`

* **Parameters:**
  * **n_conv_layers** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of convolutional layers used. Default: 15
  * **nf** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of channels per convolutional layer. Default: 64
  * **img_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of channels of your input image. Default: 1 (greyscale)
  * **residual_denoising** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to use a residual connection between input image and the network output. Default: False
  * **norm** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalization to use in the convolutional layers. Choose from instance_norm, batch_norm, or None (no norm). Default: None
  * **orthogonal_init** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Apply orthogonal initialization to the convolutional weights. Ignored if pretrained not None. Default: True
  * **last_conv_bias** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Set the learnable bias on or off on the final convolution. Default: True
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – use a pretrained network. If `pretrained=None`, the weights will be initialized
    at random (or orthogonally, see `orthogonal_init`). If `pretrained='download'`, the original FFDNet
    weights are downloaded (only available for the two initializations listed above).
    `pretrained` can also be set as a path to the user’s own pretrained weights.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details. Default: ‘download’
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to put the model on.

<hr />

* **References:**

* <a id='footcite-zhang2018ffdnet'>**[1]**</a> Kai Zhang, Wangmeng Zuo, and Lei Zhang. Ffdnet: toward a fast and flexible solution for cnn-based image denoising. *IEEE Transactions on Image Processing*, 27(9):4608–4622, 2018.

#### forward(x, sigma)

Run the denoiser on image with noise level $\sigma$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noise level. If `sigma` is a float, it is used for all images in the batch.
    If `sigma` is a tensor, it must be of shape `(batch_size,)`.
