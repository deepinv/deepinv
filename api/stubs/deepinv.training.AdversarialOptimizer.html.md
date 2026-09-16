# AdversarialOptimizer

### *class* deepinv.training.AdversarialOptimizer(optimizer_g, optimizer_d, zero_grad_g_only=False, zero_grad_d_only=False)

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Optimizer for adversarial training that encapsulates both generator and discriminator’s optimizers.

* **Parameters:**
  * **optimizer_g** ([*torch.optim.Optimizer*](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)) – generator’s torch optimizer
  * **optimizer_d** ([*torch.optim.Optimizer*](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)) – discriminator’s torch optimizer
  * **zero_grad_g_only** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to only zero_grad generator, defaults to `False`
  * **zero_grad_d_only** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to only zero_grad discriminator, defaults to `False`

#### load_state_dict(state_dict)

Load state_dict which must have “G” and “D” keys for generator and discriminator respectively

* **Parameters:**
  **state_dict** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – state_dict with keys “G” and “D”.

#### state_dict(\*args, \*\*kwargs)

Return both generator and discriminator’s state_dicts with keys “G” and “D”.

#### zero_grad(set_to_none=True)

zero_grad generator and discriminator optimizers, optionally only zero_grad one of them.

* **Parameters:**
  **set_to_none** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to set gradients to None, defaults to True

<a id="sphx-glr-backref-deepinv-training-adversarialoptimizer"></a>

## Examples using `AdversarialOptimizer`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
