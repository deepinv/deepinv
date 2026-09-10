# UAIRGeneratorLoss

### *class* deepinv.loss.adversarial.UAIRGeneratorLoss(weight_adv=0.5, weight_mc=1, metric=None, D=None, device='cpu')

Bases: [`GeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.GeneratorLoss.html.md#deepinv.loss.adversarial.GeneratorLoss)

Reimplementation of UAIR generator’s adversarial loss.

The loss, introduced by Pajot *et al.*<sup>[1](#footcite-pajot2019unsupervised)</sup>, is defined as follows, to be minimized by the generator:

$\mathcal{L}=\mathcal{L}_\text{adv}(\hat y, y;D)+\lVert \forw{\inverse{\hat y}}- \hat y\rVert^2_2,\quad\hat y=\forw{\hat x}$

where the standard adversarial loss is

$\mathcal{L}_\text{adv}(y,\hat y;D)=\mathbb{E}_{y\sim p_y}\left[q(D(y))\right]+\mathbb{E}_{\hat y\sim p_{\hat y}}\left[q(1-D(\hat y))\right]$

See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for examples of training generator and discriminator models.

Simple example (assuming a pretrained discriminator):

```default
from deepinv.models import DCGANDiscriminator
D = DCGANDiscriminator() # assume pretrained discriminator

loss = UAIRGeneratorLoss(D=D)

l = loss(y, y_hat, physics, model)

l.backward()
```

* **Parameters:**
  * **weight_adv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – weight for adversarial loss, defaults to 0.5 (from original paper)
  * **weight_mc** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – weight for measurement consistency, defaults to 1.0 (from original paper)
  * **metric** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric for measurement consistency, defaults to [`torch.nn.MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)
  * **D** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – discriminator network. If not specified, D must be provided in forward(), defaults to None.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, defaults to “cpu”

<hr />

* **References:**

* <a id='footcite-pajot2019unsupervised'>**[1]**</a> Arthur Pajot, Emmanuel De Bézenac, and Patrick Gallinari. Unsupervised adversarial image reconstruction. In *International conference on learning representations*. 2019.

#### forward(y, y_hat, physics, model, D=None, \*\*kwargs)

Forward pass for UAIR generator’s adversarial loss.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input measurement
  * **y_hat** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – re-measured reconstruction
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward physics
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – reconstruction network
  * **D** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – discriminator model. If None, then D passed from \_\_init_\_ used. Defaults to None.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-adversarial-uairgeneratorloss"></a>

## Examples using `UAIRGeneratorLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
