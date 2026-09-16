# SupAdversarialDiscriminatorLoss

### *class* deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss(weight_adv=1.0, D=None, device='cpu', \*\*kwargs)

Bases: [`DiscriminatorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.DiscriminatorLoss.html.md#deepinv.loss.adversarial.DiscriminatorLoss)

Supervised adversarial consistency loss for discriminator.

This loss was as used in conditional GANs such as Kupyn *et al.*<sup>[1](#footcite-kupyn2018deblurgan)</sup> and generative models such as Bora *et al.*<sup>[2](#footcite-bora2017compressed)</sup>.

Constructs adversarial loss between reconstructed image and the ground truth, to be maximized by discriminator.

$\mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]$

See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for examples of training generator and discriminator models.

* **Parameters:**
  * **weight_adv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – weight for adversarial loss, defaults to 1.0
  * **D** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – discriminator network. If not specified, D must be provided in forward(), defaults to None.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, defaults to “cpu”

<hr />

* **References:**

* <a id='footcite-kupyn2018deblurgan'>**[1]**</a> Orest Kupyn, Volodymyr Budzan, Mykola Mykhailych, Dmytro Mishkin, and Jiří Matas. Deblurgan: blind motion deblurring using conditional adversarial networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 8183–8192. 2018.
* <a id='footcite-bora2017compressed'>**[2]**</a> Ashish Bora, Ajil Jalal, Eric Price, and Alexandros G Dimakis. Compressed sensing using generative models. In *International conference on machine learning*, 537–546. PMLR, 2017.

#### forward(x, x_net, D=None, \*\*kwargs)

Forward pass for supervised adversarial discriminator loss.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – ground truth image
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – reconstructed image
  * **D** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – discriminator model. If None, then D passed from \_\_init_\_ used. Defaults to None.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-adversarial-supadversarialdiscriminatorloss"></a>

## Examples using `SupAdversarialDiscriminatorLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
