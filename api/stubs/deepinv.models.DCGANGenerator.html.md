# DCGANGenerator

### *class* deepinv.models.DCGANGenerator(output_size=64, nz=100, ngf=64, nc=3, dim=2)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

DCGAN Generator.

The DCGAN generator model was originally proposed by Radford *et al.*<sup>[1](#footcite-radford2015unsupervised)</sup>
and takes a latent sample as input.

Implementation taken from [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for how to use this for adversarial training.

* **Parameters:**
  * **output_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – desired square size of output image. Choose from 64 or 128, defaults to 64
  * **nz** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – latent dimension, defaults to 100
  * **ngf** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – hidden layer size, defaults to 64
  * **nc** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of image output channels, defaults to 3
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **References:**

* <a id='footcite-radford2015unsupervised'>**[1]**</a> Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. *arXiv preprint arXiv:1511.06434*, 2015.

#### forward(z, \*args, \*\*kwargs)

Generate an image

* **Parameters:**
  **z** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – latent vector

<a id="sphx-glr-backref-deepinv-models-dcgangenerator"></a>

## Examples using `DCGANGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
