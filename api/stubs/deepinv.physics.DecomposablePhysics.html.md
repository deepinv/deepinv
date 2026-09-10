# DecomposablePhysics

### *class* deepinv.physics.DecomposablePhysics(U=None, V_adjoint=None, img_size=None, U_adjoint=None, V=None, mask=1.0, device='cpu', \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Parent class for linear operators with SVD decomposition.

The singular value decomposition is expressed as

$$
A = U\text{diag}(s)V^{\top} \in \mathbb{R}^{m\times n}
$$

where $U\in\mathbb{C}^{m\times m}$ and $V\in\mathbb{C}^{n\times n}$
are orthonormal linear transformations and $\text{diag}(s)\in\mathbb{R}_{+}^{m \times n}$ is the possibly rectangular singular values matrix.

* **Parameters:**
  * **U** (*None* *|* *Callable*) – orthonormal transformation. If `None` (default), it is set to the identity function.
  * **V_adjoint** (*None* *|* *Callable*) – transpose of V. If `None` (default), it is set to the identity function.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – (optional, only required if V and/or U_adjoint are not provided) size of the signal/image `x`, e.g. `(C, ...)` where `C` is the number of channels and `...` are the spatial dimensions,
    used for the automatic adjoint computation.
  * **U_adjoint** (*None* *|* *Callable*) – transpose of U. If `None` (default), it is computed automatically using [`deepinv.physics.adjoint_function()`](https://deepinv.org/api/stubs/deepinv.physics.adjoint_function.html.md#deepinv.physics.adjoint_function)
    from the `U` function and the `img_size` parameter.
    This automatic adjoint is computed using automatic differentiation, which is slower than a closed form adjoint, and can
    have a larger memory footprint. If you want to use the automatic adjoint, you should set the `img_size` parameter.
  * **V** (*None* *|* *Callable*) – If `None` (default), it is computed automatically using [`deepinv.physics.adjoint_function()`](https://deepinv.org/api/stubs/deepinv.physics.adjoint_function.html.md#deepinv.physics.adjoint_function)
    from the `V_adjoint` function and the `img_size` parameter.
    This automatic adjoint is computed using automatic differentiation, which is slower than a closed form adjoint, and can
    have a larger memory footprint. If you want to use the automatic adjoint, you should set the `img_size` parameter.
  * **mask** ([*torch.nn.parameter.Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – Singular values of the transform
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – cpu or cuda, every registered buffer and module parameters are recursively pushed onto the device during initialization.

<hr />

* **Examples:**
  Recreation of the Inpainting operator using the DecomposablePhysics class:
  ```pycon
  >>> from deepinv.physics import DecomposablePhysics
  >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
  >>> img_size = (1, 1, 3, 3)  # Input size
  >>> mask = torch.tensor([[1, 0, 1], [1, 0, 1], [1, 0, 1]])  # Binary mask
  >>> U = lambda x: x  # U is the identity operation
  >>> U_adjoint = lambda x: x  # U_adjoint is the identity operation
  >>> V = lambda x: x  # V is the identity operation
  >>> V_adjoint = lambda x: x  # V_adjoint is the identity operation
  >>> mask_svd = mask.float().unsqueeze(0).unsqueeze(0)  # Convert the mask to torch.Tensor and adjust its dimensions
  >>> physics = DecomposablePhysics(U=U, U_adjoint=U_adjoint, V=V, V_adjoint=V_adjoint, mask=mask_svd)
  ```

  Apply the operator to a random tensor:
  ```pycon
  >>> x = torch.randn(img_size)
  >>> with torch.no_grad():
  ...     physics.A(x)  # Apply the masking
  tensor([[[[ 1.5410, -0.0000, -2.1788],
            [ 0.5684, -0.0000, -1.3986],
            [ 0.4033,  0.0000, -0.7193]]]])
  ```

#### A(x, mask=None, \*\*kwargs)

Applies the forward operator $y = A(x)$.

If a mask/singular values is provided, it is used to apply the forward operator,
and also stored as the current mask/singular values.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor
  * **mask** ([*torch.nn.parameter.Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – singular values.
* **Returns:**
  output tensor
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_A_adjoint(y, mask=None, \*\*kwargs)

A helper function that computes $A A^{\top}y$.

Using the SVD decomposition, we have $A A^{\top} = U\text{diag}(s^2)U^{\top}$.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the product $AA^{\top}y$.

#### A_adjoint(y, mask=None, \*\*kwargs)

Computes the adjoint of the forward operator $\tilde{x} = A^{\top}y$.

If a mask/singular values is provided, it is used to apply the adjoint operator,
and also stored as the current mask/singular values.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor
  * **mask** ([*torch.nn.parameter.Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – singular values.
* **Returns:**
  output tensor
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_adjoint_A(x, mask=None, \*\*kwargs)

A helper function that computes $A^{\top} A x$.

Using the SVD decomposition, we have $A^{\top}A = V\text{diag}(s^2)V^{\top}$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal/image.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the product $A^{\top}Ax$.

#### A_dagger(y, mask=None, \*\*kwargs)

Computes $A^{\dagger}y = x$ in an efficient manner leveraging the singular vector decomposition.

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – a measurement $y$ to reconstruct via the pseudoinverse.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) The reconstructed image $x$.

#### U(x)

Applies the $U$ operator of the SVD decomposition.

#### NOTE
This method should be overwritten by the user to define its custom `DecomposablePhysics` operator.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor

#### U_adjoint(x, \*\*kwargs)

Applies the $U^{\top}$ operator of the SVD decomposition.

#### NOTE
This method should be overwritten by the user to define its custom `DecomposablePhysics` operator.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor

#### V(x, \*\*kwargs)

Applies the $V$ operator of the SVD decomposition.

#### NOTE
This method should be overwritten by the user to define its custom `DecomposablePhysics` operator.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor

#### V_adjoint(x)

Applies the $V^{\top}$ operator of the SVD decomposition.

#### NOTE
This method should be overwritten by the user to define its custom `DecomposablePhysics` operator.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor

#### prox_l2(z, y, gamma, \*\*kwargs)

Computes proximal operator of $f(x)=\frac{\gamma}{2}\|Ax-y\|^2$
in an efficient manner leveraging the singular vector decomposition.

* **Parameters:**
  * **z** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – signal tensor
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements tensor
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – hyperparameter $\gamma$ of the proximal operator
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) estimated signal tensor

<a id="sphx-glr-backref-deepinv-physics-decomposablephysics"></a>

## Examples using `DecomposablePhysics`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">  <div class="sphx-glr-thumbnail-title">3D denoising</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to reconstruct a noisy and incomplete image using the deep image prior.">  <div class="sphx-glr-thumbnail-title">Reconstructing an image using the deep image prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use the expected patch log likelihood (EPLL) prior :footcitezoran2011learning. for denoising and inpainting of natural images. To this end, we consider the inverse problem y = Ax+\\epsilon, where A is either the identity (for denoising) or a masking operator (for inpainting) and \\epsilon\\sim\\mathcal{N}(0,\\sigma^2 I) is white Gaussian noise with standard deviation \\sigma.">  <div class="sphx-glr-thumbnail-title">Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise &lt;deepinv.physics.PoissonGaussianNoise&gt; using the Generalized Anscombe Transform (GAT) &lt;deepinv.models.AnscombeDenoiser&gt;, which converts any Gaussian denoiser into a Poisson-Gaussian denoiser :footcitemakitalo2012optimal.">  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding &lt;deepinv.physics.functional.liu_jia_pad&gt; :footciteliu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm :footcitekawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from :footcitezhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcitetachella2024unsure.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Deep Equilibrium Attention Least Squares (DEAL) model in DeepInverse for both denoising and a simple reconstruction settings.">  <div class="sphx-glr-thumbnail-title">DEAL denoising and reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
