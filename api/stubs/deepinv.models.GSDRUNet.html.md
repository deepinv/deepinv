# GSDRUNet

### *class* deepinv.models.GSDRUNet(alpha=1.0, in_channels=3, out_channels=3, nb=2, nc=(64, 128, 256, 512), act_mode='E', pretrained=None, device=torch.device('cpu'))

Bases:

Gradient Step Denoiser with DRUNet architecture.

Based on the GSPnP method from Hurault *et al.*<sup>[1](#footcite-hurault2021gradient)</sup>.

* **Parameters:**
  * **alpha** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Relaxation parameter
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of input channels
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of output channels
  * **nb** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of blocks in the DRUNet
  * **nc** (*Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – number of channels per convolutional layer in the DRUNet. The network has a fixed number of 4 scales with `nb` blocks per scale (default: `[64,128,256,512]`).
  * **act_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – activation mode, “R” for ReLU, “L” for LeakyReLU “E” for ELU and “S” for Softplus.
  * **downsample_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Downsampling mode, “avgpool” for average pooling, “maxpool” for max pooling, and
    “strideconv” for convolution with stride 2.
  * **upsample_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Upsampling mode, “convtranspose” for convolution transpose, “pixelshuffle” for pixel
    shuffling, and “upconv” for nearest neighbour upsampling with additional convolution.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use a pretrained network. If `pretrained=None`, the weights will be initialized at random
    using Pytorch’s default initialization. If `pretrained='download'`, the weights will be downloaded from an
    online repository (only available for the default architecture with 3 or 1 input/output channels).
    Finally, `pretrained` can also be set as a path to the user’s own pretrained weights.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – gpu or cpu.

<hr />

* **References:**

* <a id='footcite-hurault2021gradient'>**[1]**</a> Samuel Hurault, Arthur Leclaire, and Nicolas Papadakis. Gradient step denoiser for convergent plug-and-play. In *International Conference on Learning Representations*. 2021.

<a id="sphx-glr-backref-deepinv-models-gsdrunet"></a>

## Examples using `GSDRUNet`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Implementation of :footciteromano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) :footcitehurault2021gradient which provides an explicit prior.">  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
