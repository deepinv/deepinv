<a id="denoisers"></a>

# Denoisers

The [`deepinv.models.Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser) base class describe
denoisers that take a noisy image as input and return a denoised image.
They can be used as a building block for plug-and-play restoration, for building unrolled architectures,
for [artifact removal networks](https://deepinv.org/user_guide/reconstruction/deep-reconstructors.html.md#artifact), or as standalone denoisers. All denoisers have a
[`forward`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser.forward) method that takes a
noisy image and a noise level (which generally corresponds to the standard deviation of the noise)
as input and returns a denoised image:

```pycon
>>> import torch
>>> import deepinv as dinv
>>> denoiser = dinv.models.DRUNet()
>>> sigma = 0.1
>>> image = torch.ones(1, 3, 32, 32) * .5
>>> noisy_image = image + torch.randn(1, 3, 32, 32) * sigma
>>> denoised_image = denoiser(noisy_image, sigma)
```

#### NOTE
Some denoisers (e.g., [`deepinv.models.DnCNN`](https://deepinv.org/api/stubs/deepinv.models.DnCNN.html.md#deepinv.models.DnCNN)) do not use the information about the noise level.
In this case, the noise level is ignored.

<a id="deep-denoisers"></a>

## Deep denoisers

We provide the following list of deep denoising architectures,
which are based on CNN, Transformer or hybrid CNN-Transformer modules. Several denoisers accept an optional `dim` keyword at initialization, allowing to build the 2D or 3D variant.
See [Description of weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more information on pretrained denoisers.

#### Deep denoisers

| Model                                                                                                  | Type            | Tensor Size (C, H, W)        | Pretrained Weights                    | Noise level aware   | 3D variant   |
|--------------------------------------------------------------------------------------------------------|-----------------|------------------------------|---------------------------------------|---------------------|--------------|
| [`deepinv.models.AutoEncoder`](https://deepinv.org/api/stubs/deepinv.models.AutoEncoder.html.md#deepinv.models.AutoEncoder) | Fully connected | Any                          | No                                    | No                  | No           |
| [`deepinv.models.UNet`](https://deepinv.org/api/stubs/deepinv.models.UNet.html.md#deepinv.models.UNet)               | CNN             | Any C; H,W>8                 | No                                    | No                  | Yes          |
| [`deepinv.models.DnCNN`](https://deepinv.org/api/stubs/deepinv.models.DnCNN.html.md#deepinv.models.DnCNN)             | CNN             | Any C, H, W                  | RGB, grayscale                        | No                  | Yes          |
| [`deepinv.models.DRUNet`](https://deepinv.org/api/stubs/deepinv.models.DRUNet.html.md#deepinv.models.DRUNet)           | CNN-UNet        | Any C; H,W>8                 | RGB, grayscale                        | Yes                 | Yes          |
| [`deepinv.models.GSDRUNet`](https://deepinv.org/api/stubs/deepinv.models.GSDRUNet.html.md#deepinv.models.GSDRUNet)       | CNN-UNet        | Any C; H,W>8                 | RGB, grayscale                        | Yes                 | No           |
| [`deepinv.models.SCUNet`](https://deepinv.org/api/stubs/deepinv.models.SCUNet.html.md#deepinv.models.SCUNet)           | CNN-Transformer | Any C, H, W                  | No                                    | No                  | No           |
| [`deepinv.models.SwinIR`](https://deepinv.org/api/stubs/deepinv.models.SwinIR.html.md#deepinv.models.SwinIR)           | CNN-Transformer | Any C, H, W                  | RGB                                   | No                  | No           |
| [`deepinv.models.DiffUNet`](https://deepinv.org/api/stubs/deepinv.models.DiffUNet.html.md#deepinv.models.DiffUNet)       | Transformer     | Any C; H,W = 64, 128, 256, … | RGB                                   | Yes                 | No           |
| [`deepinv.models.Restormer`](https://deepinv.org/api/stubs/deepinv.models.Restormer.html.md#deepinv.models.Restormer)     | CNN-Transformer | Any C, H, W                  | RGB, grayscale, deraining, deblurring | No                  | No           |
| [`deepinv.models.ICNN`](https://deepinv.org/api/stubs/deepinv.models.ICNN.html.md#deepinv.models.ICNN)               | CNN             | Any C; H, W = 128, 256,…     | No                                    | No                  | Yes          |
| [`deepinv.models.NCSNpp`](https://deepinv.org/api/stubs/deepinv.models.NCSNpp.html.md#deepinv.models.NCSNpp)           | CNN-Transformer | Any C, H, W                  | RGB, diffusion                        | Yes                 | No           |
| [`deepinv.models.ADMUNet`](https://deepinv.org/api/stubs/deepinv.models.ADMUNet.html.md#deepinv.models.ADMUNet)         | CNN-Transformer | Any C, H, W                  | RGB, diffusion                        | Yes                 | No           |
| [`deepinv.models.DScCP`](https://deepinv.org/api/stubs/deepinv.models.DScCP.html.md#deepinv.models.DScCP)             | Unrolled        | Any C, H, W                  | RGB                                   | Yes                 | Yes          |
| [`deepinv.models.RAM`](https://deepinv.org/api/stubs/deepinv.models.RAM.html.md#deepinv.models.RAM)                 | CNN-UNet        | C=1, 2, 3; H,W>8             | C=1, 2, 3                             | Yes                 | No           |
| [`deepinv.models.FFDNet`](https://deepinv.org/api/stubs/deepinv.models.FFDNet.html.md#deepinv.models.FFDNet)           | CNN             | Any C; H,W must be even      | RGB, grayscale                        | Yes                 | No           |

<a id="non-learned-denoisers"></a>

## Classical denoisers

All denoisers in this list are non-learned (except for EPLL)
and rely on hand-crafted priors. Some of these denoisers also support 3D data,
underlined in the table below by (D) in the tensor size which accounts for depth dimension.

#### Non-Learned Denoisers Overview

| Model                                                                                                                  | Info                                                                                                              | Tensor Size (C, H, W)   |
|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------|
| [`deepinv.models.BM3D`](https://deepinv.org/api/stubs/deepinv.models.BM3D.html.md#deepinv.models.BM3D)                               | Patch-based denoiser                                                                                              | C=1 or C=3, any H, W.   |
| [`deepinv.models.BilateralFilter`](https://deepinv.org/api/stubs/deepinv.models.BilateralFilter.html.md#deepinv.models.BilateralFilter)         | Distance and range kernel-based filter                                                                            | Any C, H, W             |
| [`deepinv.models.MedianFilter`](https://deepinv.org/api/stubs/deepinv.models.MedianFilter.html.md#deepinv.models.MedianFilter)               | Non-learned filter                                                                                                | Any C, H, W             |
| [`deepinv.models.TVDenoiser`](https://deepinv.org/api/stubs/deepinv.models.TVDenoiser.html.md#deepinv.models.TVDenoiser)                   | [`Total variation prior`](https://deepinv.org/api/stubs/deepinv.optim.TVPrior.html.md#deepinv.optim.TVPrior)                      | Any C, (D), H, W        |
| [`deepinv.models.TVL1Denoiser`](https://deepinv.org/api/stubs/deepinv.models.TVL1Denoiser.html.md#deepinv.models.TVL1Denoiser)               | [`Total variation L1 prior`](https://deepinv.org/api/stubs/deepinv.optim.TVL1Prior.html.md#deepinv.optim.TVL1Prior)                 | Any C, (D), H, W        |
| [`deepinv.models.TGVDenoiser`](https://deepinv.org/api/stubs/deepinv.models.TGVDenoiser.html.md#deepinv.models.TGVDenoiser)                 | Total generalized variation prior                                                                                 | Any C, (D), H, W        |
| [`deepinv.models.WaveletDenoiser`](https://deepinv.org/api/stubs/deepinv.models.WaveletDenoiser.html.md#deepinv.models.WaveletDenoiser)         | [`Sparsity in orthogonal wavelet domain`](https://deepinv.org/api/stubs/deepinv.optim.WaveletPrior.html.md#deepinv.optim.WaveletPrior) | Any C, (D), H, W        |
| [`deepinv.models.WaveletDictDenoiser`](https://deepinv.org/api/stubs/deepinv.models.WaveletDictDenoiser.html.md#deepinv.models.WaveletDictDenoiser) | Sparsity in overcomplete wavelet domain                                                                           | Any C, (D), H, W        |
| [`deepinv.models.EPLLDenoiser`](https://deepinv.org/api/stubs/deepinv.models.EPLLDenoiser.html.md#deepinv.models.EPLLDenoiser)               | Learned patch-prior                                                                                               | C=1 or C=3, any H, W    |

<a id="model-utils"></a>

## Model Utilities

### Poisson-Gaussian denoisers

The Gaussian denoisers in the library can be turned into Poisson-Gaussian denoisers using the variance-stabilizing Anscombe transform.
The class [`deepinv.models.AnscombeDenoiser`](https://deepinv.org/api/stubs/deepinv.models.AnscombeDenoiser.html.md#deepinv.models.AnscombeDenoiser) wraps any Gaussian denoiser and applies the Anscombe transform to the input
and the inverse transform to the output of the denoiser, allowing you to use Gaussian denoisers for Poisson-Gaussian noise.
Moreover, some denoisers such as [`deepinv.models.RAM`](https://deepinv.org/api/stubs/deepinv.models.RAM.html.md#deepinv.models.RAM) are natively trained for Poisson-Gaussian noise,
and other denoisers such as [`deepinv.models.DRUNet`](https://deepinv.org/api/stubs/deepinv.models.DRUNet.html.md#deepinv.models.DRUNet) can receive spatial noise levels, allowing them to be used for Poisson-Gaussian noise as well.
See [Poisson-Gaussian Denoising with the Generalized Anscombe Transform](https://deepinv.org/auto_examples/physics/demo_anscombe.html.md#sphx-glr-auto-examples-physics-demo-anscombe-py).

### Equivariant denoisers

Denoisers can be turned into equivariant denoisers by wrapping them with the
[`deepinv.models.EquivariantDenoiser`](https://deepinv.org/api/stubs/deepinv.models.EquivariantDenoiser.html.md#deepinv.models.EquivariantDenoiser) class, which symmetrizes the denoiser
with respect to a transform from our [available transforms](https://deepinv.org/user_guide/training/transforms.html.md#transform) such as [`deepinv.transform.Rotate`](https://deepinv.org/api/stubs/deepinv.transform.Rotate.html.md#deepinv.transform.Rotate)
or [`deepinv.transform.Reflect`](https://deepinv.org/api/stubs/deepinv.transform.Reflect.html.md#deepinv.transform.Reflect). You retain full flexibility by passing in the transform of choice.
The denoising can either be averaged over the entire group of transformation (making the denoiser equivariant) or
performed on 1 or n transformations sampled uniformly at random in the group, making the denoiser a Monte-Carlo
estimator of the exact equivariant denoiser.

### Complex denoisers

Most denoisers in the library are designed to process real images. However, some problems, e.g., phase retrieval,
require processing complex-valued images. The function [`deepinv.models.complex.to_complex_denoiser`](https://deepinv.org/api/stubs/deepinv.models.complex.to_complex_denoiser.html.md#deepinv.models.complex.to_complex_denoiser) can convert any real-valued denoiser into
a complex-valued denoiser. It can be simply called by `complex_denoiser = to_complex_denoiser(denoiser)`.

### Dynamic networks

When using time-varying (i.e. dynamic) data of 5D shape (B,C,T,H,W), the reconstruction network must be adapted
using [`deepinv.models.TimeAveragingNet`](https://deepinv.org/api/stubs/deepinv.models.TimeAveragingNet.html.md#deepinv.models.TimeAveragingNet).

To adapt any existing network to take dynamic data as independent time-slices, [`deepinv.models.TimeAgnosticNet`](https://deepinv.org/api/stubs/deepinv.models.TimeAgnosticNet.html.md#deepinv.models.TimeAgnosticNet)
creates a time-agnostic wrapper that flattens the time dimension into the batch dimension.

### MMSE denoiser

The [`deepinv.models.MMSE`](https://deepinv.org/api/stubs/deepinv.models.MMSE.html.md#deepinv.models.MMSE) class implements the closed-form MMSE denoiser assuming that the prior distribution is a Dirac-mixture based on a given dataset.
This closed-form denoiser can be used to obtain a performance upper-bound on deep denoisers trained to approximate the MMSE.

<a id="model-wrappers"></a>

## Wrappers

We provide wrappers to use models from other libraries as DeepInv denoisers.

### Model from HuggingFace Diffusers

Any diffusion model from the [HuggingFace Diffusers library](https://huggingface.co/docs/diffusers/index) can be wrapped as a DeepInv denoiser
using the [`deepinv.models.DiffusersDenoiserWrapper`](https://deepinv.org/api/stubs/deepinv.models.DiffusersDenoiserWrapper.html.md#deepinv.models.DiffusersDenoiserWrapper) class. A model can be instantiated as simply as follows:

```pycon
>>> from deepinv.models import DiffusersDenoiserWrapper
>>> denoiser = DiffusersDenoiserWrapper(mode_id="google/ddpm-ema-celebahq-256")
```

It can be used as any other DeepInv denoiser `denoised_image = denoiser(noisy_image, sigma)`. It also supports conditional denoising as long as the underlying model does.
This wrapper allows you to leverage state-of-the-art diffusion models for other inverse problems beyond image generation, in particular for posterior sampling.
See [this example](https://deepinv.org/auto_examples/sampling/demo_diffusers.html.md#sphx-glr-auto-examples-sampling-demo-diffusers-py) for more details.
