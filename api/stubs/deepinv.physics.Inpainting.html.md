# Inpainting

### *class* deepinv.physics.Inpainting(img_size, mask=None, pixelwise=True, device='cpu', rng=None, \*\*kwargs)

Bases: [`DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics)

Inpainting forward operator, keeps a subset of entries.

The operator is described by the diagonal matrix

$$
A = \text{diag}(m) \in \mathbb{R}^{n\times n}
$$

where $m$ is a binary mask with $n$ entries.

This operator is linear and has a trivial SVD decomposition, which allows for fast computation
of the pseudo-inverse and proximal operator.

An existing operator can be loaded from a saved `.pth` file via `self.load_state_dict(save_path)`,
in a similar fashion to [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

Masks can also be created on-the-fly using mask generators such as
[`BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator), see example below.

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – If the input is a float, the entries of the mask will be sampled from a bernoulli
    distribution with probability equal to `mask`. If the input is a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) matching `img_size`,
    the mask will be set to this tensor. If `mask` is [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), it must be shape that is broadcastable
    to input shape and will be broadcast during forward call.
    If `None`, it must be set during forward pass or using `update` method.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – size of the input images without batch dimension e.g. of shape `(C, H, W)` or `(C, M)` or `(M,)`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – gpu or cpu.
  * **pixelwise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.
    If existing mask passed (i.e. mask is [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)), this has no effect.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – a pseudorandom random number generator for the mask generation. Default to None.

<hr />

* **Examples:**
  Inpainting operator using defined mask, removing the second column of a 3x3 image:
  ```pycon
  >>> from deepinv.physics import Inpainting
  >>> seed = torch.manual_seed(0) # Random seed for reproducibility
  >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
  >>> mask = torch.zeros(1, 3, 3) # Define empty mask
  >>> mask[:, 2, :] = 1 # Keeping last line only
  >>> physics = Inpainting(mask=mask, img_size=x.shape[1:])
  >>> physics(x)
  tensor([[[[ 0.0000, -0.0000, -0.0000],
            [ 0.0000, -0.0000, -0.0000],
            [ 0.4033,  0.8380, -0.7193]]]])
  ```

  Inpainting operator using random mask, keeping 70% of the entries of a 3x3 image:
  ```pycon
  >>> from deepinv.physics import Inpainting
  >>> seed = torch.manual_seed(0) # Random seed for reproducibility
  >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
  >>> physics = Inpainting(mask=0.7, img_size=x.shape[1:])
  >>> physics(x)
  tensor([[[[ 1.5410, -0.0000, -2.1788],
            [ 0.5684, -0.0000, -1.3986],
            [ 0.4033,  0.0000, -0.0000]]]])
  ```

  Generate random masks on-the-fly using mask generators:
  ```pycon
  >>> from deepinv.physics import Inpainting
  >>> from deepinv.physics.generator import BernoulliSplittingMaskGenerator
  >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
  >>> physics = Inpainting(img_size=x.shape[1:])
  >>> gen = BernoulliSplittingMaskGenerator(x.shape[1:], split_ratio=0.7)
  >>> params = gen.step(batch_size=1, seed = 0) # Generate random mask
  >>> physics(x, **params) # Set mask on-the-fly
  tensor([[[[-0.4033, -0.0000,  0.1820],
            [-0.8567,  1.1006, -1.0712],
            [ 0.1227, -0.0000,  0.3731]]]])
  >>> physics.update(**params) # Alternatively update mask before forward call
  >>> physics(x)
  tensor([[[[-0.4033, -0.0000,  0.1820],
            [-0.8567,  1.1006, -1.0712],
            [ 0.1227, -0.0000,  0.3731]]]])
  ```

<hr />

* **Used in benchmarks:**

- [DIV2K Inpainting easy](https://deepinv.org/auto_benchmarks/div2k_inpainting_easy.html.md#div2k-inpainting-easy)

#### \_\_mul_\_(other)

Concatenates two forward operators $A = A_1\circ A_2$ via the mul operation

If the second operator is an Inpainting or MRI operator, the masks are multiplied element-wise,
otherwise the default implementation of LinearPhysics is used (see [`deepinv.physics.LinearPhysics.__mul__()`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.__mul__)).

* **Parameters:**
  **other** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Physics operator $A_2$
* **Returns:**
  ([`deepinv.physics.Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) concatenated operator
* **Return type:**
  [*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

#### noise(x, \*\*kwargs)

Incorporates noise into the measurements $\tilde{y} = N(y)$

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – clean measurements
* **Return torch.Tensor:**
  noisy measurements
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-physics-inpainting"></a>

## Examples using `Inpainting`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to reconstruct a noisy and incomplete image using the deep image prior.">  <div class="sphx-glr-thumbnail-title">Reconstructing an image using the deep image prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use the expected patch log likelihood (EPLL) prior :footcitezoran2011learning. for denoising and inpainting of natural images. To this end, we consider the inverse problem y = Ax+\\epsilon, where A is either the identity (for denoising) or a masking operator (for inpainting) and \\epsilon\\sim\\mathcal{N}(0,\\sigma^2 I) is white Gaussian noise with standard deviation \\sigma.">  <div class="sphx-glr-thumbnail-title">Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm :footcitekawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from :footcitezhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Deep Equilibrium Attention Least Squares (DEAL) model in DeepInverse for both denoising and a simple reconstruction settings.">  <div class="sphx-glr-thumbnail-title">DEAL denoising and reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
