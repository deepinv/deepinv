# CSGMGenerator

### *class* deepinv.models.CSGMGenerator(backbone_generator=DCGANGenerator(), inf_max_iter=2500, inf_tol=1e-4, inf_lr=1e-2, inf_progress_bar=False)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Adapts a generator model backbone (e.g DCGAN) for CSGM or AmbientGAN.

This approach was proposed by Bora *et al.*<sup>[1](#footcite-bora2017compressed)</sup> and Bora *et al.*<sup>[2](#footcite-bora2018ambientgan)</sup>.

At train time, the generator samples latent vector from Unif[-1, 1] and passes through backbone.

At test time, CSGM/AmbientGAN runs an optimisation to find the best latent vector that fits the input
measurements y, then outputs the corresponding reconstruction.

This generator can be overridden for more advanced optimisation algorithms by overriding `optimize_z`.

See [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for how to use this for adversarial training.

#### NOTE
At train time, this generator discards the measurements `y`, but these measurements are used at test time.
This means that train PSNR will be meaningless but test PSNR will be correct.

* **Parameters:**
  * **backbone_generator** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – any neural network that maps a latent vector of length `nz` to an image, must have `nz` attribute. Defaults to DCGANGenerator()
  * **inf_max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum iterations at inference-time optimisation, defaults to 2500
  * **inf_tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – tolerance of inference-time optimisation, defaults to 1e-2
  * **inf_lr** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – learning rate of inference-time optimisation, defaults to 1e-2
  * **inf_progress_bar** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to display progress bar for inference-time optimisation, defaults to False

<hr />

* **References:**

* <a id='footcite-bora2017compressed'>**[1]**</a> Ashish Bora, Ajil Jalal, Eric Price, and Alexandros G Dimakis. Compressed sensing using generative models. In *International conference on machine learning*, 537–546. PMLR, 2017.
* <a id='footcite-bora2018ambientgan'>**[2]**</a> Ashish Bora, Eric Price, and Alexandros G Dimakis. Ambientgan: generative models from lossy measurements. In *International conference on learning representations*. 2018.

#### forward(y, physics, \*args, \*\*kwargs)

Forward pass of generator model.

At train time, the generator samples latent vector from Unif[-1, 1] and passes through backbone.

At test time, CSGM/AmbientGAN runs an optimisation to find the best latent vector that fits the input
measurements y, then outputs the corresponding reconstruction.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement to reconstruct
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward model

#### optimize_z(z, y, physics)

Run inference-time optimisation of latent z that is consistent with input measurement y according to physics.

The optimisation is defined with simple stopping criteria. Override this function for more advanced optimisation.

* **Parameters:**
  * **z** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – initial latent variable guess
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement with which to compare reconstructed image
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – forward model
* **Returns:**
  optimized latent z

#### random_latent(device, requires_grad=True)

Generate a latent sample to feed into generative model.

The model must have an attribute `nz` which is the latent dimension.

* **Parameters:**
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – torch device
  * **requires_grad** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to require gradient, defaults to True.

<a id="sphx-glr-backref-deepinv-models-csgmgenerator"></a>

## Examples using `CSGMGenerator`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
