# DataFidelity

### *class* deepinv.optim.DataFidelity(d=None)

Bases: [`Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential)

Base class for the data fidelity term $\distance{A(x)}{y}$ where $A$ is the forward operator,
$x\in\xset$ is a variable and $y\in\yset$ is the data, and where $d$ is a distance function,
from the class [`deepinv.optim.Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance).

* **Parameters:**
  **d** (*Callable*) – distance function $d(x, y)$ between a variable $x$ and an observation $y$.
  The distance $\distancename$ is not optional: pass a callable `d` to the constructor or use a subclass instead e.g.,
  [`deepinv.optim.L2`](https://deepinv.org/api/stubs/deepinv.optim.L2.html.md#deepinv.optim.L2).

#### bregman_prox(x, bregman_potential, y, physics, \*args, gamma=1.0, stepsize_inter=1.0, max_iter_inter=50, tol_inter=1e-3, \*\*kwargs)

(Right) Bregman proximal operator of $\gamma \datafid{x}{y}$, with Bregman potential $\phi$.

Compute the Bregman proximity operator of the fidelity term, i.e.

$$
\operatorname{prox}^\phi_{\gamma \datafidname}(x) = \underset{u}{\text{argmin}} \; \gamma \, \datafid{u}{y} + D_\phi(u,x)
$$

where $D_\phi(u,x)$ stands for the Bregman divergence with potential $\phi$.

#### WARNING
If $\phi$ is the squared Euclidean norm ([`deepinv.optim.BregmanL2`](https://deepinv.org/api/stubs/deepinv.optim.BregmanL2.html.md#deepinv.optim.BregmanL2)), this operator
reduces to the [`prox`](#deepinv.optim.DataFidelity.prox). Otherwise, this function will use a
gradient descent method to compute the Bregman proximity operator, which may be slow and not
guaranteed to converge.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **bregman_potential** ([*deepinv.optim.Bregman*](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)) – Bregman potential $\phi$ to be used in the Bregman proximity operator.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – step size for the proximity operator.
  * **stepsize_inter** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – step size for the internal optimization.
  * **max_iter_inter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations for the internal optimization.
  * **tol_inter** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – tolerance for the internal optimization.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) Bregman proximity operator $\operatorname{prox}^\phi_{\gamma \datafidname}(x)$, computed in $x$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### fn(x, y, physics, \*args, \*\*kwargs)

Computes the data fidelity term $\datafid{x}{y} = \distance{\forw{x}}{y}$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the data fidelity is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) data fidelity $\datafid{x}{y}$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, y, physics, \*args, \*\*kwargs)

Calculates the gradient of the data fidelity term $\datafidname$ at $x$.

The gradient is computed using the chain rule:

$$
\nabla_x \distance{\forw{x}}{y} = \left. \frac{\partial A}{\partial x} \right|_x^\top \nabla_u \distance{u}{y},
$$

where $\left. \frac{\partial A}{\partial x} \right|_x$ is the Jacobian of $A$ at $x$, and $\nabla_u \distance{u}{y}$ is computed using `grad_d` with $u = \forw{x}$. The multiplication is computed using the `A_vjp` method of the physics.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the gradient is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) gradient $\nabla_x \datafid{x}{y}$, computed in $x$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, y, physics, gamma=1.0, stepsize_inter=1.0, max_iter_inter=50, tol_inter=1e-3, \*\*kwargs)

Proximal operator of $\gamma \datafid{x}{y}$

Compute the proximal operator of the fidelity term $\operatorname{prox}_{\gamma \datafidname}$, i.e.

$$
\operatorname{prox}_{\gamma \datafidname}(x) = \underset{u}{\text{argmin}} \; \gamma \, d(\forw{u},y)+\frac{1}{2}\|u-x\|_2^2
$$

#### WARNING
If the proximity operator is not available in closed form, this function will use a
gradient descent method to compute the proximity operator, which may be slow and not guaranteed to converge.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – step size for the proximity operator.
  * **stepsize_inter** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – step size for the internal optimization.
  * **max_iter_inter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations for the internal optimization.
  * **tol_inter** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – tolerance for the internal optimization.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator computed in $x$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox_conjugate(x, y, physics, \*args, gamma=1.0, lamb=1.0, \*\*kwargs)

Proximal operator of the convex conjugate of $\lambda \datafidname$.

Compute $\operatorname{prox}_{\gamma (\lambda \datafidname)^*}$ using the Moreau identity

$$
\operatorname{prox}_{\gamma (\lambda \datafidname)^*}(x) = x - \gamma \operatorname{prox}_{\frac{\lambda}{\gamma} \datafidname}\left(\frac{x}{\gamma}\right)
$$

where $\operatorname{prox}_{\gamma \datafidname}$ is computed with
[`prox`](#deepinv.optim.DataFidelity.prox).

#### WARNING
The Moreau identity is only valid if the data fidelity term is convex.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Data $y$.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics model.
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – step size for the proximity operator.
  * **lamb** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – $\lambda$ parameter in front of $\datafidname$.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator $\operatorname{prox}_{\gamma (\lambda \datafidname)^*}(x)$, computed in $x$.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-datafidelity"></a>

## Examples using `DataFidelity`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from :footciteaghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a Ptychography phase retrieval operator and generate phaseless measurements from a given image.">  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we show how to use the deepinv.physics.Scattering forward model.">  <div class="sphx-glr-thumbnail-title">Inverse scattering problem</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of :footciteromano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) :footcitehurault2021gradient which provides an explicit prior.">  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from :footcitezhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm :footcitegregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
