# GaussianNoise

### *class* deepinv.physics.GaussianNoise(sigma=0.1, rng=None)

Bases: [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel)

Gaussian noise $y=z+\epsilon$ where $\epsilon\sim \mathcal{N}(0,I\sigma^2)$.

* **Parameters:**
  * **sigma** (*Union* *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – Standard deviation of the noise.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.

<hr />

* **Examples:**
  Adding gaussian noise to a physics operator by setting the `noise_model`
  attribute of the physics operator:
  ```pycon
  >>> from deepinv.physics import Denoising, GaussianNoise
  >>> import torch
  >>> physics = Denoising()
  >>> physics.noise_model = GaussianNoise()
  >>> x = torch.rand(3, 1, 2, 2)
  >>> y = physics(x)
  ```

  We can sum 2 GaussianNoise instances:
  ```pycon
  >>> gaussian_noise_1 = GaussianNoise(sigma=3.0)
  >>> gaussian_noise_2 = GaussianNoise(sigma=4.0)
  >>> gaussian_noise = gaussian_noise_1 + gaussian_noise_2
  >>> y = gaussian_noise(x)
  >>> gaussian_noise.sigma.item()
  5.0
  ```

  We can also multiply a GaussianNoise by a float:
  $scaled\_gaussian\_noise(x) = \lambda \times gaussian\_noise(x)$
  <br/>
  ```pycon
  >>> scaled_gaussian_noise = 3.0 * gaussian_noise
  >>> y = scaled_gaussian_noise(x)
  >>> scaled_gaussian_noise.sigma.item()
  15.0
  ```

  We can also create a batch of GaussianNoise with different standard deviations:
  $x=[x_1, ..., x_b]$
  <br/>
  $t=[[[[\lambda_1]]], ..., [[[\lambda_b]]]]$ a batch of scaling factors.
  <br/>
  $[t \times gaussian](x) = [\lambda_1 \times gaussian(x_1), ..., \lambda_b \times gaussian(x_b)]$
  <br/>
  ```pycon
  >>> t = torch.rand((x.size(0),) + (1,) * (x.dim() - 1)) # if x.shape = (b, 3, 32, 32) then t.shape = (b, 1, 1, 1)
  >>> batch_gaussian_noise = t * gaussian_noise
  >>> y = batch_gaussian_noise(x)
  >>> assert (t[0]*gaussian_noise).sigma.item() == batch_gaussian_noise.sigma[0].item(), "Wrong standard deviation value for the first GaussianNoise."
  ```

<hr />

* **Used in benchmarks:**

- [CBSD68 gaussian denoising](https://deepinv.org/auto_benchmarks/cbsd68_gaussian_denoising.html.md#cbsd68-gaussian-denoising)
- [DIV2K Gaussian Deblurring](https://deepinv.org/auto_benchmarks/div2k_gaussian_deblurring.html.md#div2k-gaussian-deblurring)

#### \_\_add_\_(other)

Sum of 2 gaussian noises via + operator.

$\sigma = \sqrt{\sigma_1^2 + \sigma_2^2}$

* **Parameters:**
  **other** ([*deepinv.physics.GaussianNoise*](#deepinv.physics.GaussianNoise)) – Gaussian with standard deviation $\sigma$
* **Returns:**
  ([`deepinv.physics.GaussianNoise`](#deepinv.physics.GaussianNoise)) – Gaussian noise with the sum of the linear operators.

#### \_\_mul_\_(other)

Element-wise multiplication of a GaussianNoise via `*` operator.

1. If `other` is a [`NoiseModel`](https://deepinv.org/api/stubs/deepinv.physics.NoiseModel.html.md#deepinv.physics.NoiseModel), then applies the multiplication from `NoiseModel`.
2. If `other` is a [`float`](https://docs.python.org/3.9/library/functions.html#float), then the standard deviation of the GaussianNoise is multiplied by `other`.
   > $x=[x_1, ..., x_b]$ a batch of images.
   > <br/>
   > $\lambda$ a float.
   > <br/>
   > $\sigma = [\lambda \times \sigma_1, ..., \lambda \times \sigma_b]$
   > <br/>
3. If `other` is a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), then the standard deviation of the GaussianNoise is multiplied by `other`.
   > $x=[x_1, ..., x_b]$ a batch of images.
   > <br/>
   > $other=[[[[\lambda_1]]], ..., [[[\lambda_b]]]]$ a batch of scaling factors.
   > <br/>
   > $\sigma = [\lambda \times \sigma_1, ..., \lambda \times \sigma_b]$
   > <br/>

* **Parameters:**
  **other** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *or* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Scaling factor for the GaussianNoise’s standard deviation.
* **Returns:**
  ([`deepinv.physics.GaussianNoise`](#deepinv.physics.GaussianNoise)) – A new GaussianNoise with the new standard deviation.

#### forward(x, sigma=None, seed=None, \*\*kwargs)

Adds the noise to measurements x

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – standard deviation of the noise.
    If not `None`, it will overwrite the current noise level.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.
* **Returns:**
  noisy measurements

<a id="sphx-glr-backref-deepinv-physics-gaussiannoise"></a>

## Examples using `GaussianNoise`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">  <div class="sphx-glr-thumbnail-title">3D denoising</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to reconstruct a noisy and incomplete image using the deep image prior.">  <div class="sphx-glr-thumbnail-title">Reconstructing an image using the deep image prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use the expected patch log likelihood (EPLL) prior :footcitezoran2011learning. for denoising and inpainting of natural images. To this end, we consider the inverse problem y = Ax+\\epsilon, where A is either the identity (for denoising) or a masking operator (for inpainting) and \\epsilon\\sim\\mathcal{N}(0,\\sigma^2 I) is white Gaussian noise with standard deviation \\sigma.">  <div class="sphx-glr-thumbnail-title">Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of :footciteromano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) :footcitehurault2021gradient which provides an explicit prior.">  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm :footcitekawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from :footcitezhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcitetachella2024unsure.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Deep Equilibrium Attention Least Squares (DEAL) model in DeepInverse for both denoising and a simple reconstruction settings.">  <div class="sphx-glr-thumbnail-title">DEAL denoising and reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
