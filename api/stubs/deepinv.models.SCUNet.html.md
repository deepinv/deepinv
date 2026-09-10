# SCUNet

### *class* deepinv.models.SCUNet(in_nc=3, config=(4, 4, 4, 4, 4, 4, 4), dim=64, drop_path_rate=0.0, input_resolution=256, pretrained='download', device='cpu')

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

SCUNet denoising network.

The Swin-Conv-UNet (SCUNet) denoising was introduced by Zhang *et al.*<sup>[1](#footcite-zhang2023practical)</sup>.

* **Parameters:**
  * **in_nc** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of input channels. Default: 3.
  * **config** (*Sequence*) – number of layers in each stage. Default: [4, 4, 4, 4, 4, 4, 4].
  * **dim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels in each layer. Default: 64.
  * **drop_path_rate** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – drop path per sample rate (stochastic depth) for each layer. Default: 0.0.
  * **input_resolution** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – input resolution. Default: 256.
  * **pretrained** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use a pretrained network. If `pretrained=None`, the weights will be initialized at random
    using Pytorch’s default initialization. If `pretrained='download'`, the weights will be downloaded from an
    online repository (only available for the default architecture).
    Finally, `pretrained` can also be set as a path to the user’s own pretrained weights. Default: ‘download’.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – training or testing mode. Default: False.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – gpu or cpu. Default: ‘cpu’.

#### NOTE
This class requires the `timm` package to be installed. Install with `pip install timm`.

<hr />

* **References:**

* <a id='footcite-zhang2023practical'>**[1]**</a> Kai Zhang, Yawei Li, Jingyun Liang, Jiezhang Cao, Yulun Zhang, Hao Tang, Deng-Ping Fan, Radu Timofte, and Luc Van Gool. Practical blind image denoising via swin-conv-unet and data synthesis. *Machine Intelligence Research*, 20(6):822–836, 2023.

<a id="sphx-glr-backref-deepinv-models-scunet"></a>

## Examples using `SCUNet`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of the denoisers in DeepInverse. A denoiser is a model that takes in a noisy image and outputs a denoised version of it. Basically, it solves the following problem:">  <div class="sphx-glr-thumbnail-title">Benchmarking pretrained denoisers</div>
</div>
<!-- thumbnail-parent-div-close --></div>
