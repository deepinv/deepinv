# Denoising

### *class* deepinv.physics.Denoising(noise_model=None, device='cpu', \*\*kwargs)

Bases: [`DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.md#deepinv.physics.DecomposablePhysics)

Forward operator for denoising problems.

The linear operator is just the identity mapping $A(x)=x$

* **Parameters:**
  * **noise** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – noise distribution, e.g., [`deepinv.physics.GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.md#deepinv.physics.GaussianNoise), or a user-defined torch.nn.Module. By default, it is set to Gaussian noise with a standard deviation of 0.1.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – cpu or cuda, every registered buffer and module parameters are recursively pushed onto the device during initialization.

<hr />

* **Examples:**
  Denoising operator with Gaussian noise with standard deviation 0.1:
  ```pycon
  >>> from deepinv.physics import Denoising, GaussianNoise
  >>> seed = torch.manual_seed(0) # Random seed for reproducibility
  >>> x = 0.5*torch.randn(1, 1, 3, 3) # Define random 3x3 image
  >>> physics = Denoising(GaussianNoise(sigma=0.1))
  >>> with torch.no_grad():
  ...     physics(x)
  tensor([[[[ 0.7302, -0.2064, -1.0712],
            [ 0.1985, -0.4322, -0.8064],
            [ 0.2139,  0.3624, -0.3223]]]])
  ```

<hr />

* **Used in benchmarks:**

- [CBSD68 gaussian denoising](https://deepinv.org/auto_benchmarks/cbsd68_gaussian_denoising.md#cbsd68-gaussian-denoising)

<a id="sphx-glr-backref-deepinv-physics-denoising"></a>

## Examples using `Denoising`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">![](auto_examples/metrics/images/thumb/sphx_glr_demo_test_time_tuning_thumb.png)

[Blind inverse problems with no reference metrics](https://deepinv.org/auto_examples/metrics/demo_test_time_tuning.md)

  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model terris2025reconstruct to solve inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_foundation_model_thumb.png)

[Inference and fine-tune a foundation model](https://deepinv.org/auto_examples/models/demo_foundation_model.md)

  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_3D_denoising_thumb.png)

[3D denoising of brain MRI with wavelet and TV priors](https://deepinv.org/auto_examples/optimization/demo_3D_denoising.md)

  <div class="sphx-glr-thumbnail-title">3D denoising of brain MRI with wavelet and TV priors</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use the expected patch log likelihood (EPLL) prior zoran2011learning. for denoising and inpainting of natural images. To this end, we consider the inverse problem y = Ax+\\epsilon, where A is either the identity (for denoising) or a masking operator (for inpainting) and \\epsilon\\sim\\mathcal{N}(0,\\sigma^2 I) is white Gaussian noise with standard deviation \\sigma.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_epll_thumb.png)

[Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting](https://deepinv.org/auto_examples/optimization/demo_epll.md)

  <div class="sphx-glr-thumbnail-title">Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">![](auto_examples/optimization/images/thumb/sphx_glr_demo_patch_priors_CT_thumb.png)

[Patch priors for limited-angle computed tomography](https://deepinv.org/auto_examples/optimization/demo_patch_priors_CT.md)

  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise using the Generalized Anscombe Transform (GAT), which converts any Gaussian denoiser into a Poisson-Gaussian denoiser makitalo2012optimal.">![](auto_examples/physics/images/thumb/sphx_glr_demo_anscombe_thumb.png)

[Poisson-Gaussian Denoising with the Generalized Anscombe Transform](https://deepinv.org/auto_examples/physics/demo_anscombe.md)

  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in here.">![](auto_examples/physics/images/thumb/sphx_glr_demo_physics_tour_thumb.png)

[Tour of forward sensing operators](https://deepinv.org/auto_examples/physics/demo_physics_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_lowfieldmri_thumb.png)

[Low-field MRI denoising without ground truth](https://deepinv.org/auto_examples/self-supervised-learning/demo_lowfieldmri.md)

  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_n2n_denoising_thumb.png)

[Self-supervised denoising with the Neighbor2Neighbor loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_n2n_denoising.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_poisson2sparse_thumb.png)

[Poisson denoising using Poisson2Sparse](https://deepinv.org/auto_examples/self-supervised-learning/demo_poisson2sparse.md)

  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss monroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_r2r_denoising_thumb.png)

[Self-supervised denoising with the Generalized R2R loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_r2r_denoising.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_sure_denoising_thumb.png)

[Self-supervised denoising with the SURE loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_sure_denoising.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by tachella2024unsure.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_unsure_thumb.png)

[Self-supervised denoising with the UNSURE loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_unsure.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">![](auto_examples/transforms-equivariance/images/thumb/sphx_glr_demo_transforms_thumb.png)

[Image transforms for equivariance & augmentations](https://deepinv.org/auto_examples/transforms-equivariance/demo_transforms.md)

  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Deep Equilibrium Attention Least Squares (DEAL) model in DeepInverse for both denoising and a simple reconstruction settings.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_deal_thumb.png)

[DEAL denoising and reconstruction](https://deepinv.org/auto_examples/unfolded/demo_deal.md)

  <div class="sphx-glr-thumbnail-title">DEAL denoising and reconstruction</div>
</div>
<!-- thumbnail-parent-div-close --></div>
