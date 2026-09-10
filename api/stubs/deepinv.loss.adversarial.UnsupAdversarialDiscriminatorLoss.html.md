# UnsupAdversarialDiscriminatorLoss

### *class* deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss(weight_adv=1.0, D=None, device='cpu')

Bases: [`DiscriminatorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.DiscriminatorLoss.html.md#deepinv.loss.adversarial.DiscriminatorLoss)

Unsupervised adversarial consistency loss for discriminator.

This loss was used for unsupervised generative models such as in Bora *et al.*<sup>[1](#footcite-bora2018ambientgan)</sup>.

Constructs adversarial loss between input measurement and re-measured reconstruction, to be maximized
by discriminator.

$\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]$

See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for examples of training generator and discriminator models.

* **Parameters:**
  * **weight_adv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – weight for adversarial loss, defaults to 1.0
  * **D** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – discriminator network. If not specified, D must be provided in forward(), defaults to None.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, defaults to “cpu”

<hr />

* **References:**

* <a id='footcite-bora2018ambientgan'>**[1]**</a> Ashish Bora, Eric Price, and Alexandros G Dimakis. Ambientgan: generative models from lossy measurements. In *International conference on learning representations*. 2018.

#### forward(y, y_hat, D=None, \*\*kwargs)

Forward pass for unsupervised adversarial discriminator loss.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurement
  * **y_hat** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – re-measured reconstruction
  * **D** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – discriminator model. If None, then D passed from \_\_init_\_ used. Defaults to None.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-adversarial-unsupadversarialdiscriminatorloss"></a>

## Examples using `UnsupAdversarialDiscriminatorLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
