# gaussian_blur

### deepinv.physics.functional.gaussian_blur(psf_size=None, sigma=(1.0, 1.0), angle=0.0, device='cpu', dtype=torch.float32)

Creates a batch of N-dimensional anisotropic Gaussian kernels (1D, 2D, or 3D) with independent sigma ranges and rotation angles for each kernel.

The dimension (1D, 2D, or 3D) of the kernel is determined by the length of the `psf_size` tuple, if `None`, it is determined by the length of the `sigma` tuple. If both are provided, their dimensions must match. Batch size is determined by the shape of `sigma` or `angle`, if both are provided, their shapes must be compatible. If both are scalars (int, float or a tuple), the batch size is 1 and the same kernel is repeated for the entire batch. If `sigma` is a tensor of shape (batch_size, dim), then each kernel in the batch can have different sigma values.

* **Parameters:**
  * **psf_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*) – Shape of the point spread function for the batch, it can be a tuple specifying the size for each dimension (e.g., (7, 3) for 2D), or None to automatically determine the size based on sigma.
  * **sigma** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]*  *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – If a float or integer is provided, the kernel is isotropic with the specified sigma for all dimensions. The user can define an anisotropic kernel by passing a tuple of float. If a float, integer, or tuple is provided, the same kernel is repeated for the entire batch. If a tensor of shape (batch_size, dim) is provided, each kernel in the batch can have different sigma values.
  * **angle** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]*  *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Rotation angle(s) for each kernel in the batch (degrees). For 2D kernels, this is a single angle of rotation in the plane. For 3D kernels, this can be a tuple of three angles (gamma, beta, alpha) representing rotations around the x, y, and z axes respectively. In 3D, if a single angle is provided, it is applied as a rotation around the z-axis. If a tuple of three angles is provided, they are applied as rotations around the x, y, and z axes in that order. If a tensor of shape (batch_size,) for 2D or (batch_size, 3) for 3D is provided, each kernel in the batch can have its own rotation angle(s).
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to create the tensor on.
* **Returns:**
  A tensor of batched N-dimensional Gaussian kernels of shape `(batch_size, *psf_size)`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

## Examples using `gaussian_blur`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">![](auto_examples/basics/images/thumb/sphx_glr_demo_pretrained_model_thumb.png)

[Use a pretrained model](https://deepinv.org/auto_examples/basics/demo_pretrained_model.md)

  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">![](auto_examples/basics/images/thumb/sphx_glr_demo_quickstart_thumb.png)

[5 minute quickstart tutorial](https://deepinv.org/auto_examples/basics/demo_quickstart.md)

  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example introduces the Richardson-Lucy algorithm for deconvolution in the blind setting, where both the underlying clean image and the blur kernel are unknown.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_richardsonlucy_thumb.png)

[Blind Richardson-Lucy deconvolution](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_richardsonlucy.md)

  <div class="sphx-glr-thumbnail-title">Blind Richardson-Lucy deconvolution</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">![](auto_examples/distributed/images/thumb/sphx_glr_demo_unrolled_distributed_thumb.png)

[Distributed Training of Unfolded Networks](https://deepinv.org/auto_examples/distributed/demo_unrolled_distributed.md)

  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">![](auto_examples/metrics/images/thumb/sphx_glr_demo_test_time_tuning_thumb.png)

[Blind inverse problems with no reference metrics](https://deepinv.org/auto_examples/metrics/demo_test_time_tuning.md)

  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_custom_prior_thumb.png)

[Image deblurring with custom deep explicit prior.](https://deepinv.org/auto_examples/optimization/demo_custom_prior.md)

  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_multispectral_demosaicing_thumb.png)

[Multispectral demosaicing from raw sensor data](https://deepinv.org/auto_examples/optimization/demo_multispectral_demosaicing.md)

  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm sheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting richardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_poisson_mlem_thumb.png)

[Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)](https://deepinv.org/auto_examples/optimization/demo_poisson_mlem.md)

  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">![](auto_examples/physics/images/thumb/sphx_glr_demo_blur_tour_thumb.png)

[Tour of blur operators](https://deepinv.org/auto_examples/physics/demo_blur_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding liu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">![](auto_examples/physics/images/thumb/sphx_glr_demo_liu_jia_padding_thumb.png)

[Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding](https://deepinv.org/auto_examples/physics/demo_liu_jia_padding.md)

  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in here.">![](auto_examples/physics/images/thumb/sphx_glr_demo_physics_tour_thumb.png)

[Tour of forward sensing operators](https://deepinv.org/auto_examples/physics/demo_physics_tour.md)

  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spatial_unwrapping_thumb.png)

[Spatial unwrapping and modulo imaging](https://deepinv.org/auto_examples/physics/demo_spatial_unwrapping.md)

  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See hurault2023convergent for more details.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_mirror_descent_thumb.png)

[Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_mirror_descent.md)

  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver to compute the proximal step w.r.t. the data-fidelity term (e.g., ADMM or HQS):  ">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_unfolded_constant_memory_thumb.png)

[Reducing the memory and computational complexity of unfolded network training](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constant_memory.md)

  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div>
<!-- thumbnail-parent-div-close --></div>
