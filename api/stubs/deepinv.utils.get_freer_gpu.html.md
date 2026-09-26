# get_freer_gpu

### deepinv.utils.get_freer_gpu(verbose=True, use_torch_api=True, hide_warnings=False)

Returns the GPU device with the most free memory.

Use in conjunction with `torch.cuda.is_available()`.

If `use_torch_api=True` then attempts to select GPU using only torch commands, otherwise
uses system driver to detect GPUs (via `nvidia-smi` command). The first method may be slower
but is more reliable as the former depends on environment settings.
If system method is chosen and fails, the call falls back to using torch commands and a warning
is printed. If no CUDA devices are detected, then `None` is returned.

* **Parameters:**
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – print selected GPU index and memory
  * **use_torch_api** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use torch commands if True, or Nvidia driver otherwise
  * **hide_warnings** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – suppress all warnings for all methods
* **Return torch.device device:**
  selected cuda device

#### WARNING
GPU indices in `nvidia-smi` may not match those in PyTorch if in your environment `CUDA_DEVICE_ORDER`
is not set to `PCI_BUS_ID`:
[https://discuss.pytorch.org/t/gpu-devices-nvidia-smi-and-cuda-get-device-name-output-appear-inconsistent/13150](https://discuss.pytorch.org/t/gpu-devices-nvidia-smi-and-cuda-get-device-name-output-appear-inconsistent/13150)
If the variable is not set or has different value, the call to will print a warning
(if not suppressed with `hide_warnings=True`) but will not change the device.

## Examples using `get_freer_gpu`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper carbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_deblurring_thumb.png)

[Blind deblurring with kernel estimation network](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_deblurring.html.md)

  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example focuses on blind image Gaussian denoising, i.e. the problem">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_denoising_thumb.png)

[Blind denoising with noise level estimation](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_denoising.html.md)

  <div class="sphx-glr-thumbnail-title">Blind denoising with noise level estimation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example introduces the Richardson-Lucy algorithm for deconvolution in the blind setting, where both the underlying clean image and the blur kernel are unknown.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_richardsonlucy_thumb.png)

[Blind Richardson-Lucy deconvolution](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_richardsonlucy.html.md)

  <div class="sphx-glr-thumbnail-title">Blind Richardson-Lucy deconvolution</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_connect_spyrit_thumb.png)

[Single-pixel imaging with Spyrit](https://deepinv.org/auto_examples/external-libraries/demo_connect_spyrit.html.md)

  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm sheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting richardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_poisson_mlem_thumb.png)

[Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)](https://deepinv.org/auto_examples/optimization/demo_poisson_mlem.html.md)

  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding liu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">![](auto_examples/physics/images/thumb/sphx_glr_demo_liu_jia_padding_thumb.png)

[Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding](https://deepinv.org/auto_examples/physics/demo_liu_jia_padding.html.md)

  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we show how to use the deepinv.physics.Scattering forward model.">![](auto_examples/physics/images/thumb/sphx_glr_demo_scattering_thumb.png)

[Inverse scattering problem](https://deepinv.org/auto_examples/physics/demo_scattering.html.md)

  <div class="sphx-glr-thumbnail-title">Inverse scattering problem</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_multiscale_thumb.png)

[Multi-scale Plug-and-Play for Inpainting](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_multiscale.html.md)

  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only sechaud26Equivariant.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_equivariant_splitting_thumb.png)

[Self-supervised learning with Equivariant Splitting](https://deepinv.org/auto_examples/self-supervised-learning/demo_equivariant_splitting.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_poisson2sparse_thumb.png)

[Poisson denoising using Poisson2Sparse](https://deepinv.org/auto_examples/self-supervised-learning/demo_poisson2sparse.html.md)

  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver to compute the proximal step w.r.t. the data-fidelity term (e.g., ADMM or HQS):  ">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_unfolded_constant_memory_thumb.png)

[Reducing the memory and computational complexity of unfolded network training](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constant_memory.html.md)

  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div>
<!-- thumbnail-parent-div-close --></div>
