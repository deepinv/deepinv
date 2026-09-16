# UNet

### *class* deepinv.models.UNet(in_channels=1, out_channels=1, residual=True, circular_padding=False, cat=True, bias=True, batch_norm=True, scales=None, channels_per_scale=None, device='cpu', dim=2)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

U-Net convolutional denoiser.

This network is a fully convolutional denoiser based on the U-Net architecture. The number of stages in the network is
controlled by `scales`. The width of each stage is controlled by `channels_per_scale`,
which gives the number of feature maps at each stage, from shallow to deeper stages.
The number of trainable parameters increases with both `scales` and the values in `channels_per_scale`.

If `scales` is not given, it is inferred from `channels_per_scale`. If both are omitted, defaults to
`channels_per_scale=[64, 128, 256, 512]`. If only `scales` is specified, `channels_per_scale=[64 * (2**k) for k in range(scales)]`.
When both are specified, `scales` must match the length of `channels_per_scale`.

#### WARNING
When using the bias-free batch norm via `batch_norm="biasfree"`, NaNs may be encountered
during training, causing the whole training procedure to fail.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – input image channels
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – output image channels
  * **residual** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use a skip-connection between input and output.
  * **circular_padding** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – circular padding for the convolutional layers.
  * **cat** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use skip-connections between intermediate levels.
  * **bias** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use learnable biases in conv and norm layers.
  * **batch_norm** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – if False, disable normalization entirely, if `True`, use batch normalization,
    if `batch_norm="biasfree"`, use the bias-free batch norm from Mohan *et al.*<sup>[1](#footcite-mohan2020robust)</sup>.
  * **scales** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of stages.
  * **channels_per_scale** (*Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Number of feature maps at each stage (from shallow to deep).
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to put the model on.
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **References:**

* <a id='footcite-mohan2020robust'>**[1]**</a> Sreyas Mohan, Zahra Kadkhodaie, Eero P Simoncelli, and Carlos Fernandez-Granda. Robust and interpretable blind image denoising via bias-free convolutional neural networks. In *8th International Conference on Learning Representations, ICLR 2020*. 2020.

#### forward(x, sigma=None, \*\*kwargs)

Run the denoiser on noisy image. The noise level is not used in this denoiser.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level (not used).

<a id="sphx-glr-backref-deepinv-models-unet"></a>

## Examples using `UNet`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcitetachella2024unsure.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
