# PatchGANDiscriminator

### *class* deepinv.models.PatchGANDiscriminator(input_nc=3, ndf=64, n_layers=3, use_sigmoid=False, batch_norm=True, bias=True, dim=2)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

PatchGAN Discriminator model.

This discriminator model was originally proposed by Isola *et al.*<sup>[1](#footcite-isola2017image)</sup> and classifies whether each patch of an image is real
or fake.

Implementation adapted from Kupyn *et al.*<sup>[2](#footcite-kupyn2018deblurgan)</sup>.

See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for how to use this for adversarial training.

* **Parameters:**
  * **input_nc** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of input channels, defaults to 3
  * **ndf** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – hidden layer size, defaults to 64
  * **n_layers** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of hidden conv layers, defaults to 3
  * **use_sigmoid** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use sigmoid activation at end, defaults to False
  * **batch_norm** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use batch norm layers, defaults to True
  * **bias** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use bias in conv layers, defaults to True
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **References:**

* <a id='footcite-isola2017image'>**[1]**</a> Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 1125–1134. 2017.
* <a id='footcite-kupyn2018deblurgan'>**[2]**</a> Orest Kupyn, Volodymyr Budzan, Mykola Mykhailych, Dmytro Mishkin, and Jiří Matas. Deblurgan: blind motion deblurring using conditional adversarial networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 8183–8192. 2018.

#### forward(x)

Forward pass of discriminator model.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image

<a id="sphx-glr-backref-deepinv-models-patchgandiscriminator"></a>

## Examples using `PatchGANDiscriminator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
