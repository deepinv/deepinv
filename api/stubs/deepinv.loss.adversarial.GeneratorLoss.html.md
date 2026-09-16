# GeneratorLoss

### *class* deepinv.loss.adversarial.GeneratorLoss(weight_adv=1.0, D=None, device='cpu', \*\*kwargs)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Base generator adversarial loss.

Override the forward function to call [`adversarial_loss`](#deepinv.loss.adversarial.GeneratorLoss.adversarial_loss)
with quantities depending on your specific GAN model.
For examples, see [`SupAdversarialGeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.SupAdversarialGeneratorLoss.html.md#deepinv.loss.adversarial.SupAdversarialGeneratorLoss)
and [`UnsupAdversarialGeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss.html.md#deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss)

See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for formula.

* **Parameters:**
  * **weight_adv** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – weight for adversarial loss, defaults to 1.
  * **D** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – discriminator network. If not specified, `D` must be provided in forward(), defaults to None.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – torch device, defaults to `"cpu"`

#### adversarial_loss(real, fake, D=None)

Typical adversarial loss in GAN generators.

* **Parameters:**
  * **real** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – image labelled as real, typically one originating from training set
  * **fake** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – image labelled as fake, typically a reconstructed image
  * **D** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – discriminator/critic/classifier model. If `None`, then `D` passed from `__init__` used.
    Defaults to `None`.
* **Returns:**
  generator adversarial loss
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-loss-adversarial-generatorloss"></a>

## Examples using `GeneratorLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
