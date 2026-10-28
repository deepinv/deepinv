# Prior

### *class* deepinv.optim.Prior(g=None, \*args, \*\*kwargs)

Bases: [`Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.md#deepinv.optim.Potential)

Prior term $\reg{x}$.

This is the base class for the prior term $\reg{x}$. As a child class from the Potential class, it comes with methods for computing
$\operatorname{prox}_{g}$ and $\nabla \regname$.
To implement a custom prior, for an explicit prior, overwrite $\regname$ (do not forget to specify
`self.explicit_prior = True`)

This base class is also used to implement implicit priors. For instance, in PnP methods, the method computing the
proximity operator is overwritten by a method performing denoising. For an implicit prior, overwrite `grad`
or `prox`.

#### NOTE
The methods for computing the proximity operator and the gradient of the prior rely on automatic
differentiation. These methods should not be used when the prior is not differentiable, although they will
not raise an error.

* **Parameters:**
  **g** (*Callable*) – Prior function $g(x)$.

<a id="sphx-glr-backref-deepinv-optim-prior"></a>

## Examples using `Prior`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_optim_thumb.png)

[Use iterative reconstruction algorithms](https://deepinv.org/auto_examples/basics/demo_custom_optim.md)

  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example introduces the Richardson-Lucy algorithm for deconvolution in the blind setting, where both the underlying clean image and the blur kernel are unknown.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_richardsonlucy_thumb.png)

[Blind Richardson-Lucy deconvolution](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_richardsonlucy.md)

  <div class="sphx-glr-thumbnail-title">Blind Richardson-Lucy deconvolution</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">![](auto_examples/distributed/images/thumb/sphx_glr_demo_pnp_distributed_thumb.png)

[Distributed Plug-and-Play (PnP) Reconstruction](https://deepinv.org/auto_examples/distributed/demo_pnp_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">![](auto_examples/distributed/images/thumb/sphx_glr_demo_unrolled_distributed_thumb.png)

[Distributed Training of Unfolded Networks](https://deepinv.org/auto_examples/distributed/demo_unrolled_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_tomography_thumb.png)

[Low-dose CT with ASTRA backend and Total-Variation (TV) prior](https://deepinv.org/auto_examples/external-libraries/demo_astra_tomography.md)

  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_connect_spyrit_thumb.png)

[Single-pixel imaging with Spyrit](https://deepinv.org/auto_examples/external-libraries/demo_connect_spyrit.md)

  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from aghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_ri_basic_thumb.png)

[Radio interferometric imaging with deepinverse](https://deepinv.org/auto_examples/external-libraries/demo_ri_basic.md)

  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs real prospectively undersampled multicoil brain k-space from yu2022validation.">![](auto_examples/models/images/thumb/sphx_glr_demo_prospective_mri_thumb.png)

[Reconstruct prospectively-undersampled raw multicoil MRI](https://deepinv.org/auto_examples/models/demo_prospective_mri.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct prospectively-undersampled raw multicoil MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_TV_minimisation_thumb.png)

[Image deblurring with Total-Variation (TV) prior](https://deepinv.org/auto_examples/optimization/demo_TV_minimisation.md)

  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_custom_prior_thumb.png)

[Image deblurring with custom deep explicit prior.](https://deepinv.org/auto_examples/optimization/demo_custom_prior.md)

  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_multispectral_demosaicing_thumb.png)

[Multispectral demosaicing from raw sensor data](https://deepinv.org/auto_examples/optimization/demo_multispectral_demosaicing.md)

  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">![](auto_examples/optimization/images/thumb/sphx_glr_demo_patch_priors_CT_thumb.png)

[Patch priors for limited-angle computed tomography](https://deepinv.org/auto_examples/optimization/demo_patch_priors_CT.md)

  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm sheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting richardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_poisson_mlem_thumb.png)

[Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)](https://deepinv.org/auto_examples/optimization/demo_poisson_mlem.md)

  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_wavelet_prior_thumb.png)

[Image inpainting with wavelet prior](https://deepinv.org/auto_examples/optimization/demo_wavelet_prior.md)

  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a 2D slice from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. The slice contains five hot lesions, we simulate a sinogram with deepinv.physics.PET and we compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_2d_thumb.png)

[2D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_2d.md)

  <div class="sphx-glr-thumbnail-title">2D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a volume from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. We compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_3d_thumb.png)

[3D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_3d.md)

  <div class="sphx-glr-thumbnail-title">3D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">![](auto_examples/physics/images/thumb/sphx_glr_demo_phase_retrieval_thumb.png)

[Random phase retrieval and reconstruction methods.](https://deepinv.org/auto_examples/physics/demo_phase_retrieval.md)

  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spatial_unwrapping_thumb.png)

[Spatial unwrapping and modulo imaging](https://deepinv.org/auto_examples/physics/demo_spatial_unwrapping.md)

  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spc_thumb.png)

[Pattern Ordering in a Compressive Single Pixel Camera](https://deepinv.org/auto_examples/physics/demo_spc.md)

  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in zhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_DPIR_deblur_thumb.png)

[DPIR method for PnP image deblurring.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_DPIR_deblur.md)

  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_custom_optim_thumb.png)

[PnP with custom optimization algorithm (Primal-Dual Condat-Vu)](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_custom_optim.md)

  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See hurault2023convergent for more details.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_mirror_descent_thumb.png)

[Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_mirror_descent.md)

  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_multiscale_thumb.png)

[Multi-scale Plug-and-Play for Inpainting](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_multiscale.md)

  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of romano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) hurault2021gradient which provides an explicit prior.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_RED_GSPnP_SR_thumb.png)

[Regularization by Denoising (RED) for Super-Resolution.](https://deepinv.org/auto_examples/plug-and-play/demo_RED_GSPnP_SR.md)

  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use deepinv.physics.UltrasoundPlaneWave to reconstructs an in-vivo carotid acquisition of the EPFL LTS5 ultrafast ultrasound dataset from raw RF ultrasound data.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_ultrasound_invivo_PnP_thumb.png)

[In-vivo ultrafast ultrasound reconstruction with Plug-and-Play](https://deepinv.org/auto_examples/plug-and-play/demo_ultrasound_invivo_PnP.md)

  <div class="sphx-glr-thumbnail-title">In-vivo ultrafast ultrasound reconstruction with Plug-and-Play</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_vanilla_PnP_thumb.png)

[Vanilla PnP for computed tomography (CT).](https://deepinv.org/auto_examples/plug-and-play/demo_vanilla_PnP.md)

  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_custom_kernel_thumb.png)

[Building your custom MCMC sampling algorithm.](https://deepinv.org/auto_examples/sampling/demo_custom_kernel.md)

  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_sampling_thumb.png)

[Uncertainty quantification with PnP-ULA.](https://deepinv.org/auto_examples/sampling/demo_sampling.md)

  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_DEQ_thumb.png)

[Deep Equilibrium (DEQ) algorithms for image deblurring](https://deepinv.org/auto_examples/unfolded/demo_DEQ.md)

  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm gregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_LISTA_thumb.png)

[Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing](https://deepinv.org/auto_examples/unfolded/demo_LISTA.md)

  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_custom_prior_unfolded_thumb.png)

[Learned iterative custom prior](https://deepinv.org/auto_examples/unfolded/demo_custom_prior_unfolded.md)

  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_learned_primal_dual_thumb.png)

[Learned Primal-Dual algorithm for CT scan.](https://deepinv.org/auto_examples/unfolded/demo_learned_primal_dual.md)

  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver to compute the proximal step w.r.t. the data-fidelity term (e.g., ADMM or HQS):  ">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_unfolded_constant_memory_thumb.png)

[Reducing the memory and computational complexity of unfolded network training](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constant_memory.md)

  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_unfolded_constrained_LISTA_thumb.png)

[Unfolded Chambolle-Pock for constrained image inpainting](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constrained_LISTA.md)

  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_vanilla_unfolded_thumb.png)

[Vanilla Unfolded algorithm for super-resolution](https://deepinv.org/auto_examples/unfolded/demo_vanilla_unfolded.md)

  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
