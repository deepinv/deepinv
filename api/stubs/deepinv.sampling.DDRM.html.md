# DDRM

### *class* deepinv.sampling.DDRM(denoiser, sigmas=None, eta=0.85, etab=1.0, verbose=False, eps=1e-6)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Denoising Diffusion Restoration Models (DDRM).

This class implements the Denoising Diffusion Restoration Model (DDRM) described in Kawar *et al.*<sup>[1](#footcite-kawar2022denoising)</sup>.

The DDRM is a sampling method that uses a denoiser to sample from the posterior distribution of the inverse problem.

It requires that the physics operator has a singular value decomposition, i.e.,
it is [`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics) class.

* **Parameters:**
  * **denoiser** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – a denoiser model that can handle different noise levels.
  * **sigmas** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – a list of noise levels to use in the diffusion, they should be in decreasing
    order from 1 to 0. Defaults to `np.linspace(1, 0, 100)`, i.e., 100 equally spaced noise levels from 1 to 0.
  * **eta** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – hyperparameter
  * **etab** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – hyperparameter
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, print progress

<hr />

* **Examples:**
  Denoising diffusion restoration model using a pretrained DRUNet denoiser:

```default
import deepinv as dinv
device = dinv.utils.get_device(verbose=False)
seed = torch.manual_seed(0) # Random seed for reproducibility
seed = torch.cuda.manual_seed(0) # Random seed for reproducibility on GPU
x = 0.5 * torch.ones(1, 3, 32, 32, device=device) # Define plain gray 32x32 image
physics = dinv.physics.Inpainting(
   mask=0.5, img_size=(3, 32, 32),
   noise_model=dinv.physics.GaussianNoise(0.1),
   device=device,
)
y = physics(x) # measurements
denoiser = dinv.models.DRUNet(pretrained="download").to(device)
model = dinv.sampling.DDRM(denoiser=denoiser, sigmas=np.linspace(1, 0, 10), verbose=True) # define the DDRM model
xhat = model(y, physics) # sample from the posterior distribution
(dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x)).cpu() # tensor([True])
```

<hr />

* **References:**

* <a id='footcite-kawar2022denoising'>**[1]**</a> Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. *Advances in Neural Information Processing Systems*, 35:23593–23606, 2022.

#### forward(y, physics, seed=None)

Runs the diffusion to obtain a random sample of the posterior distribution.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the measurements.
  * **physics** ([*deepinv.physics.DecomposablePhysics*](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics)) – the physics operator, which must have a singular value
    decomposition.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.

<a id="sphx-glr-backref-deepinv-sampling-ddrm"></a>

## Examples using `DDRM`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm :footcitekawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div>
<!-- thumbnail-parent-div-close --></div>
