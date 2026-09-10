# PnP

### *class* deepinv.optim.PnP(denoiser, \*args, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

Plug-and-play prior $\operatorname{prox}_{\gamma \regname}(x) = \operatorname{D}_{\sigma}(x)$.

* **Parameters:**
  **denoiser** (*Callable*) – Denoiser $\operatorname{D}_{\sigma}$.

#### prox(x, sigma_denoiser, \*args, \*\*kwargs)

Uses denoising as the proximity operator of the PnP prior $\regname$ at $x$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **sigma_denoiser** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level parameter of the denoiser.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator at $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-pnp"></a>

## Examples using `PnP`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
