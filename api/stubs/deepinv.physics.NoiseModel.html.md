# NoiseModel

### *class* deepinv.physics.NoiseModel(noise_model=None, rng=None)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class for noise model.

Noise models can be combined via [`deepinv.physics.NoiseModel.__mul__()`](#deepinv.physics.NoiseModel.__mul__).

* **Parameters:**
  * **noise_model** (*Callable*) – noise model function $N(y)$.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) *,* *None*) – (optional) a pseudorandom random number generator for the parameter generation.
    If provided, it should be on the same device as the input.

#### \_\_mul_\_(other)

Concatenates two noise $N = N_1 \circ N_2$ via the mul operation

The resulting operator will add the noise from both noise models and keep the `rng` of $N_1$.

* **Parameters:**
  **other** ([*deepinv.physics.NoiseModel*](#deepinv.physics.NoiseModel)) – Physics operator $A_2$
* **Returns:**
  (deepinv.physics.NoiseModel) concatenated operator

#### forward(input, seed=None)

Add noise to the input

* **Parameters:**
  * **input** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator.

#### rand_like(input, seed=None)

Equivalent to `torch.rand_like` but supports a pseudorandom number generator argument.

* **Parameters:**
  **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.

#### randn_like(input, seed=None)

Equivalent to `torch.randn_like` but supports a pseudorandom number generator argument.

* **Parameters:**
  **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed for the random number generator, if `rng` is provided.

#### reset_rng()

Reset the random number generator to its initial state.

#### rng_manual_seed(seed=None)

Sets the seed for the random number generator.

* **Parameters:**
  **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the seed to set for the random number generator.
  If not provided, the current state of the random number generator is used.
  .. note:: The seed will be ignored if the random number generator is not initialized.

#### update_parameters(\*\*kwargs)

Update the parameters of the noise model.

* **Parameters:**
  **kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – dictionary of parameters to update.

<a id="sphx-glr-backref-deepinv-physics-noisemodel"></a>

## Examples using `NoiseModel`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_physics_thumb.png)

[Bring your own physics](https://deepinv.org/auto_examples/basics/demo_custom_physics.html.md)

  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">![](auto_examples/basics/images/thumb/sphx_glr_demo_pretrained_model_thumb.png)

[Use a pretrained model](https://deepinv.org/auto_examples/basics/demo_pretrained_model.html.md)

  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">![](auto_examples/basics/images/thumb/sphx_glr_demo_quickstart_thumb.png)

[5 minute quickstart tutorial](https://deepinv.org/auto_examples/basics/demo_quickstart.html.md)

  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example introduces the Richardson-Lucy algorithm for deconvolution in the blind setting, where both the underlying clean image and the blur kernel are unknown.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_richardsonlucy_thumb.png)

[Blind Richardson-Lucy deconvolution](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_richardsonlucy.html.md)

  <div class="sphx-glr-thumbnail-title">Blind Richardson-Lucy deconvolution</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_optimizing_physics_parameter_thumb.png)

[Calibrating physics operators](https://deepinv.org/auto_examples/blind-inverse-problems/demo_optimizing_physics_parameter.html.md)

  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">![](auto_examples/distributed/images/thumb/sphx_glr_demo_pnp_distributed_thumb.png)

[Distributed Plug-and-Play (PnP) Reconstruction](https://deepinv.org/auto_examples/distributed/demo_pnp_distributed.html.md)

  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">![](auto_examples/distributed/images/thumb/sphx_glr_demo_unrolled_distributed_thumb.png)

[Distributed Training of Unfolded Networks](https://deepinv.org/auto_examples/distributed/demo_unrolled_distributed.html.md)

  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="The data is taken from the 2DeteCT benchmark kiss2025benchmarking and dataset kiss20232detect, which is an industrial CT dataset of various materials acquired using a proprietary scanner from CWI (i.e. sinogram-to-image). The setup is matched exactly to kiss2025benchmarking, such that you can compare DeepInverse image reconstruction methods with the values reported in kiss2025benchmarking.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_2detect_thumb.png)

[Reconstruct real CT sinograms with the 2DeteCT benchmark](https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct real CT sinograms with the 2DeteCT benchmark</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_tomography_thumb.png)

[Low-dose CT with ASTRA backend and Total-Variation (TV) prior](https://deepinv.org/auto_examples/external-libraries/demo_astra_tomography.html.md)

  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs raw non-Cartesian multicoil kspace data from the FastMRI breast dataset solomonFastMRI2025, for mammography.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_mrinufft_breast_thumb.png)

[Reconstruct accelerated non-Cartesian breast MRI acquisition data](https://deepinv.org/auto_examples/external-libraries/demo_mrinufft_breast.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct accelerated non-Cartesian breast MRI acquisition data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">![](auto_examples/metrics/images/thumb/sphx_glr_demo_test_time_tuning_thumb.png)

[Blind inverse problems with no reference metrics](https://deepinv.org/auto_examples/metrics/demo_test_time_tuning.html.md)

  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model terris2025reconstruct to solve inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_foundation_model_thumb.png)

[Inference and fine-tune a foundation model](https://deepinv.org/auto_examples/models/demo_foundation_model.html.md)

  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs real prospectively undersampled multicoil brain k-space from yu2022validation.">![](auto_examples/models/images/thumb/sphx_glr_demo_prospective_mri_thumb.png)

[Reconstruct prospectively-undersampled raw multicoil MRI](https://deepinv.org/auto_examples/models/demo_prospective_mri.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct prospectively-undersampled raw multicoil MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_3D_denoising_thumb.png)

[3D denoising of brain MRI with wavelet and TV priors](https://deepinv.org/auto_examples/optimization/demo_3D_denoising.html.md)

  <div class="sphx-glr-thumbnail-title">3D denoising of brain MRI with wavelet and TV priors</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_TV_minimisation_thumb.png)

[Image deblurring with Total-Variation (TV) prior](https://deepinv.org/auto_examples/optimization/demo_TV_minimisation.html.md)

  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_custom_prior_thumb.png)

[Image deblurring with custom deep explicit prior.](https://deepinv.org/auto_examples/optimization/demo_custom_prior.html.md)

  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to reconstruct a noisy and incomplete image using the deep image prior.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_dip_thumb.png)

[Reconstructing an image using the deep image prior.](https://deepinv.org/auto_examples/optimization/demo_dip.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstructing an image using the deep image prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use the expected patch log likelihood (EPLL) prior zoran2011learning. for denoising and inpainting of natural images. To this end, we consider the inverse problem y = Ax+\\epsilon, where A is either the identity (for denoising) or a masking operator (for inpainting) and \\epsilon\\sim\\mathcal{N}(0,\\sigma^2 I) is white Gaussian noise with standard deviation \\sigma.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_epll_thumb.png)

[Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting](https://deepinv.org/auto_examples/optimization/demo_epll.html.md)

  <div class="sphx-glr-thumbnail-title">Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">![](auto_examples/optimization/images/thumb/sphx_glr_demo_patch_priors_CT_thumb.png)

[Patch priors for limited-angle computed tomography](https://deepinv.org/auto_examples/optimization/demo_patch_priors_CT.html.md)

  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm sheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting richardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_poisson_mlem_thumb.png)

[Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)](https://deepinv.org/auto_examples/optimization/demo_poisson_mlem.html.md)

  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_wavelet_prior_thumb.png)

[Image inpainting with wavelet prior](https://deepinv.org/auto_examples/optimization/demo_wavelet_prior.html.md)

  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise using the Generalized Anscombe Transform (GAT), which converts any Gaussian denoiser into a Poisson-Gaussian denoiser makitalo2012optimal.">![](auto_examples/physics/images/thumb/sphx_glr_demo_anscombe_thumb.png)

[Poisson-Gaussian Denoising with the Generalized Anscombe Transform](https://deepinv.org/auto_examples/physics/demo_anscombe.html.md)

  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in here.">![](auto_examples/physics/images/thumb/sphx_glr_demo_physics_tour_thumb.png)

[Tour of forward sensing operators](https://deepinv.org/auto_examples/physics/demo_physics_tour.html.md)

  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spatial_unwrapping_thumb.png)

[Spatial unwrapping and modulo imaging](https://deepinv.org/auto_examples/physics/demo_spatial_unwrapping.html.md)

  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spc_thumb.png)

[Pattern Ordering in a Compressive Single Pixel Camera](https://deepinv.org/auto_examples/physics/demo_spc.html.md)

  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in zhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_DPIR_deblur_thumb.png)

[DPIR method for PnP image deblurring.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_DPIR_deblur.html.md)

  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_custom_optim_thumb.png)

[PnP with custom optimization algorithm (Primal-Dual Condat-Vu)](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_custom_optim.html.md)

  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See hurault2023convergent for more details.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_mirror_descent_thumb.png)

[Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_mirror_descent.html.md)

  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_multiscale_thumb.png)

[Multi-scale Plug-and-Play for Inpainting](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_multiscale.html.md)

  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of romano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) hurault2021gradient which provides an explicit prior.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_RED_GSPnP_SR_thumb.png)

[Regularization by Denoising (RED) for Super-Resolution.](https://deepinv.org/auto_examples/plug-and-play/demo_RED_GSPnP_SR.html.md)

  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_vanilla_PnP_thumb.png)

[Vanilla PnP for computed tomography (CT).](https://deepinv.org/auto_examples/plug-and-play/demo_vanilla_PnP.html.md)

  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_custom_kernel_thumb.png)

[Building your custom MCMC sampling algorithm.](https://deepinv.org/auto_examples/sampling/demo_custom_kernel.html.md)

  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm kawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_ddrm_thumb.png)

[Image reconstruction with a diffusion model](https://deepinv.org/auto_examples/sampling/demo_ddrm.html.md)

  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from zhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffpir_thumb.png)

[Implementing DiffPIR](https://deepinv.org/auto_examples/sampling/demo_diffpir.html.md)

  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusers_thumb.png)

[Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse](https://deepinv.org/auto_examples/sampling/demo_diffusers.html.md)

  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in chung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_dps_thumb.png)

[DPS – Posterior Sampling with Diffusion Models](https://deepinv.org/auto_examples/sampling/demo_dps.html.md)

  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">![](auto_examples/sampling/images/thumb/sphx_glr_demo_flow_matching_thumb.png)

[Flow-Matching for posterior sampling and unconditional generation](https://deepinv.org/auto_examples/sampling/demo_flow_matching.html.md)

  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.html.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_sampling_thumb.png)

[Uncertainty quantification with PnP-ULA.](https://deepinv.org/auto_examples/sampling/demo_sampling.html.md)

  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_lowfieldmri_thumb.png)

[Low-field MRI denoising without ground truth](https://deepinv.org/auto_examples/self-supervised-learning/demo_lowfieldmri.html.md)

  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_n2n_denoising_thumb.png)

[Self-supervised denoising with the Neighbor2Neighbor loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_n2n_denoising.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_poisson2sparse_thumb.png)

[Poisson denoising using Poisson2Sparse](https://deepinv.org/auto_examples/self-supervised-learning/demo_poisson2sparse.html.md)

  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss monroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_r2r_denoising_thumb.png)

[Self-supervised denoising with the Generalized R2R loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_r2r_denoising.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_scan_specific_thumb.png)

[Scan-specific zero-shot SSDU for MRI](https://deepinv.org/auto_examples/self-supervised-learning/demo_scan_specific.html.md)

  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse hendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU yaman2020self.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_splitting_loss_thumb.png)

[Self-supervised learning with measurement splitting](https://deepinv.org/auto_examples/self-supervised-learning/demo_splitting_loss.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_sure_denoising_thumb.png)

[Self-supervised denoising with the SURE loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_sure_denoising.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by tachella2024unsure.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_unsure_thumb.png)

[Self-supervised denoising with the UNSURE loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_unsure.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">![](auto_examples/transforms-equivariance/images/thumb/sphx_glr_demo_transforms_thumb.png)

[Image transforms for equivariance & augmentations](https://deepinv.org/auto_examples/transforms-equivariance/demo_transforms.html.md)

  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_DEQ_thumb.png)

[Deep Equilibrium (DEQ) algorithms for image deblurring](https://deepinv.org/auto_examples/unfolded/demo_DEQ.html.md)

  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Deep Equilibrium Attention Least Squares (DEAL) model in DeepInverse for both denoising and a simple reconstruction settings.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_deal_thumb.png)

[DEAL denoising and reconstruction](https://deepinv.org/auto_examples/unfolded/demo_deal.html.md)

  <div class="sphx-glr-thumbnail-title">DEAL denoising and reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_learned_primal_dual_thumb.png)

[Learned Primal-Dual algorithm for CT scan.](https://deepinv.org/auto_examples/unfolded/demo_learned_primal_dual.html.md)

  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver to compute the proximal step w.r.t. the data-fidelity term (e.g., ADMM or HQS):  ">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_unfolded_constant_memory_thumb.png)

[Reducing the memory and computational complexity of unfolded network training](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constant_memory.html.md)

  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_vanilla_unfolded_thumb.png)

[Vanilla Unfolded algorithm for super-resolution](https://deepinv.org/auto_examples/unfolded/demo_vanilla_unfolded.html.md)

  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
