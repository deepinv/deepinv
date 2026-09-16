# DPIR

### *class* deepinv.optim.DPIR(sigma=0.1, denoiser=None, device='cpu')

Bases: [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)

Deep Plug-and-Play (DPIR) algorithm for image restoration.

The method is based on half-quadratic splitting (HQS) and a PnP prior with a pretrained denoiser [`deepinv.models.DRUNet`](https://deepinv.org/api/stubs/deepinv.models.DRUNet.html.md#deepinv.models.DRUNet).
The optimization is stopped early and the noise level for the denoiser is adapted at each iteration.
See [DPIR method for PnP image deblurring.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_DPIR_deblur.html.md#sphx-glr-auto-examples-plug-and-play-demo-pnp-dpir-deblur-py) for more details on the implementation,
and how to adapt it to your specific problem.

This method uses a standard $\ell_2$ data fidelity term.

The DPIR method is described in Zhang *et al.*<sup>[1](#footcite-zhang2021plug)</sup>.

* **Parameters:**
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Standard deviation of the measurement noise, which controls the choice of the
    rest of the hyperparameters of the algorithm. Default is `0.1`.
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – optional denoiser. If `None`, use a pretrained denoiser [`deepinv.models.DRUNet`](https://deepinv.org/api/stubs/deepinv.models.DRUNet.html.md#deepinv.models.DRUNet).
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device to run the algorithm, either “cpu” or “cuda”. Default is “cpu”.

<hr />

* **References:**

* <a id='footcite-zhang2021plug'>**[1]**</a> Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte. Plug-and-play image restoration with deep denoiser prior. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(10):6360–6376, 2021.

<a id="sphx-glr-backref-deepinv-optim-dpir"></a>

## Examples using `DPIR`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
