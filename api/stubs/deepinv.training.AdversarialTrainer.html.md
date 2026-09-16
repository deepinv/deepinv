# AdversarialTrainer

### *class* deepinv.training.AdversarialTrainer(model, physics, optimizer, train_dataloader, losses_d, D, step_ratio_D, ...)

Bases: [`Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer)

Trainer class for training a reconstruction network using adversarial learning.

It overrides the [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) class to provide the same functionality,
whilst supporting training using adversarial losses. Note that the forward pass remains the same.

The usual reconstruction model corresponds to the generator model in an adversarial framework,
which is trained using losses specified in the `losses` argument.
Additionally, a discriminator model `D` is also jointly trained using the losses provided in `losses_d`.
The adversarial losses themselves are defined in the [Adversarial Learning](https://deepinv.org/user_guide/training/loss.html.md#adversarial-losses) module.
Examples of discriminators are in [Adversarial Networks](https://deepinv.org/user_guide/reconstruction/adversarial.html.md#adversarial).

See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for usage.

<hr />

* **Examples:**
  A very basic example:
  ```pycon
  >>> from deepinv.training import AdversarialTrainer, AdversarialOptimizer
  >>> from deepinv.loss.adversarial import SupAdversarialGeneratorLoss, SupAdversarialDiscriminatorLoss
  >>> from deepinv.models import UNet, PatchGANDiscriminator
  >>> from deepinv.physics import LinearPhysics
  >>> from deepinv.datasets.utils import PlaceholderDataset
  >>>
  >>> generator = UNet(scales=2)
  >>> discrimin = PatchGANDiscriminator(1, 2, 1)
  >>>
  >>> optimizer = AdversarialOptimizer(
  ...     torch.optim.Adam(generator.parameters()),
  ...     torch.optim.Adam(discrimin.parameters()),
  ... )
  >>>
  >>> trainer = AdversarialTrainer(
  ...     model = generator,
  ...     D = discrimin,
  ...     physics = LinearPhysics(),
  ...     train_dataloader = torch.utils.data.DataLoader(PlaceholderDataset()),
  ...     epochs = 1,
  ...     losses = SupAdversarialGeneratorLoss(),
  ...     losses_d = SupAdversarialDiscriminatorLoss(),
  ...     optimizer = optimizer,
  ...     verbose = False,
  ...     device = "cpu",
  ...     optimizer_step_multi_dataset = False
  ... )
  >>>
  >>> generator = trainer.train()
  ```

Note that this forward pass also computes `y_hat` ahead of time to avoid having to compute it multiple times,
but this is completely optional.

See [`deepinv.Trainer`](https://deepinv.org/api/stubs/deepinv.Trainer.html.md#deepinv.Trainer) for additional parameters.

#### WARNING
The multi-dataset option is not available yet when using an adversarial trainer. The `optimizer_step_multi_dataset` parameter is therefore automatically set to `False` if not set to `False` by the user.

* **Parameters:**
  * **optimizer** ([*deepinv.training.AdversarialOptimizer*](https://deepinv.org/api/stubs/deepinv.training.AdversarialOptimizer.html.md#deepinv.training.AdversarialOptimizer)) – optimizer encapsulating both generator and discriminator optimizers
  * **losses_d** ([*Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – losses to train the discriminator, e.g. adversarial losses
  * **D** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – discriminator/critic/classification model, which must take in an image and return a scalar
  * **step_ratio_D** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – every iteration, train D this many times, allowing for imbalanced generator/discriminator training. Defaults to 1.

#### check_clip_grad_D()

Check the discriminator’s gradient norm and perform gradient clipping if necessary.

Analogous to `check_clip_grad` for generator.

#### compute_loss(physics, x, y, train=True, epoch=None, step=True)

Compute losses and perform backward passes for both generator and discriminator networks.

* **Parameters:**
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Current physics operator.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Ground truth.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurement.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the model is trained, otherwise it is evaluated.
  * **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – current epoch.
  * **step** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, perform an optimization step on all datasets before optimizer step.
* **Returns:**
  (tuple) The network reconstruction x_net (for plotting and computing metrics) and
  the logs (for printing the training progress).

#### save_model(epoch, eval_psnr=None)

Save discriminator model parameters alongside other models.

#### setup_train(\*\*kwargs)

After usual Trainer setup, setup losses for discriminator too.

<a id="sphx-glr-backref-deepinv-training-adversarialtrainer"></a>

## Examples using `AdversarialTrainer`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
