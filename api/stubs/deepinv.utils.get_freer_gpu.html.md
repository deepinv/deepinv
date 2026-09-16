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
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example focuses on blind image Gaussian denoising, i.e. the problem">  <div class="sphx-glr-thumbnail-title">Blind denoising with noise level estimation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding &lt;deepinv.physics.functional.liu_jia_pad&gt; :footciteliu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we show how to use the deepinv.physics.Scattering forward model.">  <div class="sphx-glr-thumbnail-title">Inverse scattering problem</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div>
<!-- thumbnail-parent-div-close --></div>
