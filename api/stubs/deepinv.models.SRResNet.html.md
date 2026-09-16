# SRResNet

### *class* deepinv.models.SRResNet(num_blocks=16, im_c=3, feats=64, upscale=4, actv=nn.PReLU, norm='batch_norm', final_kernel_size=9, final_relu=False, pretrained=None, device=torch.device('cpu'))

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

SRResNet super-resolution network.

Convolutional super-resolution architecture introduced in Ledig *et al.*<sup>[1](#footcite-ledig2017photo)</sup>
as the generator of SRGAN. The network applies a feature-extraction conv, a stack of
residual blocks (Conv-Norm-Activation-Conv-Norm with an additive skip), a long skip
connection from the feature-extraction output, and finally a sequence of
[`torch.nn.PixelShuffle`](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)-based upsampling stages followed by a wide output
convolution.

The total upsampling factor is `upscale` and must be a power of two; the network
contains $\log_2(\text{upscale})$ upsampling stages, each doubling the spatial
resolution.

The model is registered as a [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor):
its `forward` takes a low-resolution measurement `y` and returns a
high-resolution estimate. The `physics` argument is accepted for API compatibility
but is not used by this network.

#### NOTE
The defaults correspond to the network configuration in Ledig *et al.*<sup>[1](#footcite-ledig2017photo)</sup>.

#### NOTE
Pretrained weights are available for the default RGB 4× configuration trained on
DIV2K under L1 loss with [`DownsamplingMatlab`](https://deepinv.org/api/stubs/deepinv.physics.DownsamplingMatlab.html.md#deepinv.physics.DownsamplingMatlab) (bicubic,
factor 4). These weights require `final_relu=True`. Load with
`pretrained="download"`.

* **Parameters:**
  * **num_blocks** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of residual blocks in the trunk. Default: 16
  * **im_c** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of image channels (used for both input and output). Default: 3
  * **feats** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of feature channels in the trunk. Default: 64
  * **upscale** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – upsampling factor. Must be a power of two. Default: 4
  * **actv** ([*type*](https://docs.python.org/3.9/library/functions.html#type) *[*[*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) *]*) – activation layer class, instantiated with no
    arguments. Default: [`torch.nn.ReLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU).
  * **norm** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalization layer, can be one of (‘instance_norm’, ‘batch_norm’, ‘layer_norm’, None). Default ‘batch_norm’.
  * **final_kernel_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – kernel size of the final output convolution. Must be odd. Default: 9.
  * **final_relu** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – enforce non-negativity of output by performing a relu after final conv. Default: False
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – load pretrained weights. If `"download"`, weights are
    downloaded from an online repository. If a file path string, weights are loaded
    from that path. If `None`, weights are randomly initialised. The available
    pretrained weights require the default architecture with `final_relu=True`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to put the model on. Default: ‘cpu’

<hr />

* **References:**

* <a id='footcite-ledig2017photo'>**[1]**</a> Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and others. Photo-realistic single image super-resolution using a generative adversarial network. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 4681–4690. 2017.

#### forward(y, physics=None, \*\*kwargs)

Apply the super-resolution network to a low-resolution input.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – low-resolution input image, of shape `(B, im_c, H, W)`.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward operator (not used).
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) high-resolution estimate, of shape
  `(B, im_c, upscale * H, upscale * W)`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-models-srresnet"></a>

## Examples using `SRResNet`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div>
<!-- thumbnail-parent-div-close --></div>
