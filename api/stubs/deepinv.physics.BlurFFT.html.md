# BlurFFT

### *class* deepinv.physics.BlurFFT(img_size, filter=None, device='cpu', \*\*kwargs)

Bases: [`DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics)

FFT-based blur operator.

It performs the operation

$$
y = w*x
$$

where $*$ denotes convolution and $w$ is a filter.

Blur operator based on `torch.fft` operations, which assumes a circular padding of the input, and allows for
the singular value decomposition via `deepinv.Physics.DecomposablePhysics` and has fast pseudo-inverse and prox operators.

#### WARNING
The FFT computations can lead to small numerical errors, which may result in negative values in the output even when the input is non-negative.
If used in combination with [`deepinv.physics.PoissonNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonNoise.html.md#deepinv.physics.PoissonNoise), it is recommended to set `clip_positive=True` in the noise model to avoid runtime errors.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Input image size in the form `(C, H, W)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – torch.Tensor of size `(1, c, h, w)` containing the blur filter with h<=H, w<=W and c=1 or c=C e.g.,
    [`deepinv.physics.functional.gaussian_blur()`](https://deepinv.org/api/stubs/deepinv.physics.functional.gaussian_blur.html.md#deepinv.physics.functional.gaussian_blur). If `None`, a filter must be passed to the physics before calling it . (default is `None`)
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device on which the physics’ buffers will be created. If a buffer is updated via `physics.update_parameters()`, if not None, it will be automatically casted to the device of the replaced buffer, else, use the device of the provided value. To change the device of all buffers, please use `physics.to(device)`.

<hr />

* **Examples:**
  BlurFFT operator with a basic averaging filter applied to a 16x16 black image with
  a single white pixel in the center:
  ```pycon
  >>> from deepinv.physics import BlurFFT
  >>> x = torch.zeros((1, 1, 16, 16)) # Define black image of size 16x16
  >>> x[:, :, 8, 8] = 1 # Define one white pixel in the middle
  >>> filter = torch.ones((1, 1, 2, 2)) / 4 # Basic 2x2 filter
  >>> physics = BlurFFT(filter=filter, img_size=(1, 16, 16))
  >>> y = physics(x)
  >>> y[y<1e-5] = 0.
  >>> y[:, :, 7:10, 7:10] # Display the center of the blurred image
  tensor([[[[0.2500, 0.2500, 0.0000],
            [0.2500, 0.2500, 0.0000],
            [0.0000, 0.0000, 0.0000]]]])
  ```

#### *static* get_filter_parameters(img_size, filter, device='cpu')

Create filter parameters for BlurFFT operator.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – size of the input image `(C, H, W)`.
  * **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Filter to be applied to the input image.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device where the filter tensor will be created.

#### update_parameters(filter=None, \*\*kwargs)

Updates the current filter.

* **Parameters:**
  **filter** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – New filter to be applied to the input image.

<a id="sphx-glr-backref-deepinv-physics-blurfft"></a>

## Examples using `BlurFFT`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding &lt;deepinv.physics.functional.liu_jia_pad&gt; :footciteliu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div>
<!-- thumbnail-parent-div-close --></div>
