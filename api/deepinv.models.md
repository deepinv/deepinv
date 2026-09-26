# deepinv.models

This module contains a collection of models for denoising and reconstruction.
Please refer to the [user guide](https://deepinv.org/user_guide.md#user-guide) for more information.

## Base Classes

**User Guide:** refer to [Introduction](https://deepinv.org/user_guide/reconstruction/introduction.md#reconstructors) for more information.

| [`deepinv.models.Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.md#deepinv.models.Denoiser)           | Base class for denoiser models.       |
|------------------------------------------------------------------------------------------------------------|---------------------------------------|
| [`deepinv.models.Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.md#deepinv.models.Reconstructor) | Base class for reconstruction models. |

## Classical Denoisers

**User Guide:** refer to [Classical denoisers](https://deepinv.org/user_guide/reconstruction/denoisers.md#non-learned-denoisers) for more information.

| [`deepinv.models.BM3D`](https://deepinv.org/api/stubs/deepinv.models.BM3D.md#deepinv.models.BM3D)                               | BM3D denoiser.                                                                                |
|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| [`deepinv.models.BilateralFilter`](https://deepinv.org/api/stubs/deepinv.models.BilateralFilter.md#deepinv.models.BilateralFilter)         | Bilateral filter.                                                                             |
| [`deepinv.models.MedianFilter`](https://deepinv.org/api/stubs/deepinv.models.MedianFilter.md#deepinv.models.MedianFilter)               | Median filter.                                                                                |
| [`deepinv.models.TVDenoiser`](https://deepinv.org/api/stubs/deepinv.models.TVDenoiser.md#deepinv.models.TVDenoiser)                   | Proximal operator of the isotropic Total Variation operator.                                  |
| [`deepinv.models.TVL1Denoiser`](https://deepinv.org/api/stubs/deepinv.models.TVL1Denoiser.md#deepinv.models.TVL1Denoiser)               | Compute the proximal operator of the conjugate TV-L1 regularization term.                     |
| [`deepinv.models.TGVDenoiser`](https://deepinv.org/api/stubs/deepinv.models.TGVDenoiser.md#deepinv.models.TGVDenoiser)                 | Proximal operator of (2nd order) Total Generalized Variation operator.                        |
| [`deepinv.models.WaveletDenoiser`](https://deepinv.org/api/stubs/deepinv.models.WaveletDenoiser.md#deepinv.models.WaveletDenoiser)         | Orthogonal Wavelet denoising with the $\ell_1$ norm.                                          |
| [`deepinv.models.WaveletDictDenoiser`](https://deepinv.org/api/stubs/deepinv.models.WaveletDictDenoiser.md#deepinv.models.WaveletDictDenoiser) | Overcomplete Wavelet denoising with the $\ell_1$ norm.                                        |
| [`deepinv.models.EPLLDenoiser`](https://deepinv.org/api/stubs/deepinv.models.EPLLDenoiser.md#deepinv.models.EPLLDenoiser)               | Expected Patch Log Likelihood denoising method.                                               |
| [`deepinv.models.MMSE`](https://deepinv.org/api/stubs/deepinv.models.MMSE.md#deepinv.models.MMSE)                               | Closed-form MMSE denoiser for a Dirac-mixture prior based on a given dataset of images $x_k$. |

## Deep Architectures

**User Guide:** refer to [Deep denoisers](https://deepinv.org/user_guide/reconstruction/denoisers.md#deep-denoisers) for more information.

**User Guide:** refer to [Deep Reconstruction Models](https://deepinv.org/user_guide/reconstruction/deep-reconstructors.md#deep-reconstructors) for more information.

| [`deepinv.models.AutoEncoder`](https://deepinv.org/api/stubs/deepinv.models.AutoEncoder.md#deepinv.models.AutoEncoder)         | Simple fully connected autoencoder network.                           |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`deepinv.models.UNet`](https://deepinv.org/api/stubs/deepinv.models.UNet.md#deepinv.models.UNet)                       | U-Net convolutional denoiser.                                         |
| [`deepinv.models.DnCNN`](https://deepinv.org/api/stubs/deepinv.models.DnCNN.md#deepinv.models.DnCNN)                     | DnCNN convolutional denoiser.                                         |
| [`deepinv.models.DRUNet`](https://deepinv.org/api/stubs/deepinv.models.DRUNet.md#deepinv.models.DRUNet)                   | DRUNet denoiser network.                                              |
| [`deepinv.models.SCUNet`](https://deepinv.org/api/stubs/deepinv.models.SCUNet.md#deepinv.models.SCUNet)                   | SCUNet denoising network.                                             |
| [`deepinv.models.GSDRUNet`](https://deepinv.org/api/stubs/deepinv.models.GSDRUNet.md#deepinv.models.GSDRUNet)               | Gradient Step Denoiser with DRUNet architecture.                      |
| [`deepinv.models.SwinIR`](https://deepinv.org/api/stubs/deepinv.models.SwinIR.md#deepinv.models.SwinIR)                   | SwinIR denoising network.                                             |
| [`deepinv.models.PromptIR`](https://deepinv.org/api/stubs/deepinv.models.PromptIR.md#deepinv.models.PromptIR)               | PromptIR restoration model.                                           |
| [`deepinv.models.DiffUNet`](https://deepinv.org/api/stubs/deepinv.models.DiffUNet.md#deepinv.models.DiffUNet)               | Diffusion UNet model.                                                 |
| [`deepinv.models.Restormer`](https://deepinv.org/api/stubs/deepinv.models.Restormer.md#deepinv.models.Restormer)             | Restormer denoiser network.                                           |
| [`deepinv.models.ICNN`](https://deepinv.org/api/stubs/deepinv.models.ICNN.md#deepinv.models.ICNN)                       | Convolutional Input Convex Neural Network (ICNN).                     |
| [`deepinv.models.VarNet`](https://deepinv.org/api/stubs/deepinv.models.VarNet.md#deepinv.models.VarNet)                   | VarNet or E2E-VarNet model.                                           |
| [`deepinv.models.MoDL`](https://deepinv.org/api/stubs/deepinv.models.MoDL.md#deepinv.models.MoDL)                       | MoDL unfolded network.                                                |
| [`deepinv.models.DIRECTModel`](https://deepinv.org/api/stubs/deepinv.models.DIRECTModel.md#deepinv.models.DIRECTModel)         | Pretrained MRI reconstruction models from DIRECT library.             |
| [`deepinv.models.PanNet`](https://deepinv.org/api/stubs/deepinv.models.PanNet.md#deepinv.models.PanNet)                   | PanNet architecture for pan-sharpening.                               |
| [`deepinv.models.ADMUNet`](https://deepinv.org/api/stubs/deepinv.models.ADMUNet.md#deepinv.models.ADMUNet)                 | Implementation of the ADM UNet diffusion model.                       |
| [`deepinv.models.NCSNpp`](https://deepinv.org/api/stubs/deepinv.models.NCSNpp.md#deepinv.models.NCSNpp)                   | Implementation of the DDPM++ and NCSN++ UNet architectures.           |
| [`deepinv.models.DScCP`](https://deepinv.org/api/stubs/deepinv.models.DScCP.md#deepinv.models.DScCP)                     | DScCP denoiser network.                                               |
| [`deepinv.models.RAM`](https://deepinv.org/api/stubs/deepinv.models.RAM.md#deepinv.models.RAM)                         | Reconstruct Anything Model (RAM) foundation model.                    |
| [`deepinv.models.DEAL`](https://deepinv.org/api/stubs/deepinv.models.DEAL.md#deepinv.models.DEAL)                       | Deep Equilibrium Attention Least Squares (DEAL) reconstruction model. |
| [`deepinv.models.ArtifactRemoval`](https://deepinv.org/api/stubs/deepinv.models.ArtifactRemoval.md#deepinv.models.ArtifactRemoval) | Artifact removal architecture.                                        |
| [`deepinv.models.SRResNet`](https://deepinv.org/api/stubs/deepinv.models.SRResNet.md#deepinv.models.SRResNet)               | SRResNet super-resolution network.                                    |
| [`deepinv.models.FFDNet`](https://deepinv.org/api/stubs/deepinv.models.FFDNet.md#deepinv.models.FFDNet)                   | FFDNet denoiser network.                                              |

## Model Utils

**User Guide:** refer to [Model Utilities](https://deepinv.org/user_guide/reconstruction/denoisers.md#model-utils) for more information.

| [`deepinv.models.EquivariantDenoiser`](https://deepinv.org/api/stubs/deepinv.models.EquivariantDenoiser.md#deepinv.models.EquivariantDenoiser)           | Turns the input denoiser into an equivariant denoiser with respect to geometric transforms.                                                                                                                                                                                                                                      |
|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.models.EquivariantReconstructor`](https://deepinv.org/api/stubs/deepinv.models.EquivariantReconstructor.md#deepinv.models.EquivariantReconstructor) | Equivariant reconstructor                                                                                                                                                                                                                                                                                                        |
| [`deepinv.models.TimeAgnosticNet`](https://deepinv.org/api/stubs/deepinv.models.TimeAgnosticNet.md#deepinv.models.TimeAgnosticNet)                   | Time-agnostic network wrapper.                                                                                                                                                                                                                                                                                                   |
| [`deepinv.models.TimeAveragingNet`](https://deepinv.org/api/stubs/deepinv.models.TimeAveragingNet.md#deepinv.models.TimeAveragingNet)                 | Time-averaging network wrapper.                                                                                                                                                                                                                                                                                                  |
| [`deepinv.models.Client`](https://deepinv.org/api/stubs/deepinv.models.Client.md#deepinv.models.Client)                                     | DeepInverse model API Client.                                                                                                                                                                                                                                                                                                    |
| [`deepinv.models.AnscombeDenoiser`](https://deepinv.org/api/stubs/deepinv.models.AnscombeDenoiser.md#deepinv.models.AnscombeDenoiser)                 | Wraps a Gaussian denoiser into a Poisson-Gaussian denoiser using the [`Generalized Anscombe Transform (GAT)`](https://deepinv.org/api/stubs/deepinv.models.generalized_anscombe_transform.md#deepinv.models.generalized_anscombe_transform) and its [`inverse`](https://deepinv.org/api/stubs/deepinv.models.inverse_generalized_anscombe_transform.md#deepinv.models.inverse_generalized_anscombe_transform). |

| [`deepinv.models.complex.to_complex_denoiser`](https://deepinv.org/api/stubs/deepinv.models.complex.to_complex_denoiser.md#deepinv.models.complex.to_complex_denoiser)                       | Converts a denoiser with real inputs into the one with complex inputs.   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`deepinv.models.generalized_anscombe_transform`](https://deepinv.org/api/stubs/deepinv.models.generalized_anscombe_transform.md#deepinv.models.generalized_anscombe_transform)                 | Generalized Anscombe Transform (GAT)                                     |
| [`deepinv.models.inverse_generalized_anscombe_transform`](https://deepinv.org/api/stubs/deepinv.models.inverse_generalized_anscombe_transform.md#deepinv.models.inverse_generalized_anscombe_transform) | Inverse Generalized Anscombe Transform (IGAT)                            |

## Wrappers

**User Guide:** refer to [Wrappers](https://deepinv.org/user_guide/reconstruction/denoisers.md#model-wrappers) for more information.

| [`deepinv.models.ScoreModelWrapper`](https://deepinv.org/api/stubs/deepinv.models.ScoreModelWrapper.md#deepinv.models.ScoreModelWrapper)                   | Wraps a score model as a DeepInv Denoiser.                                                                                                                   |
|--------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.models.DiffusersDenoiserWrapper`](https://deepinv.org/api/stubs/deepinv.models.DiffusersDenoiserWrapper.md#deepinv.models.DiffusersDenoiserWrapper)     | Wraps a [HuggingFace diffusers](https://huggingface.co/docs/diffusers/index) model as a DeepInv Denoiser.                                                    |
| [`deepinv.models.ComplexDenoiserWrapper`](https://deepinv.org/api/stubs/deepinv.models.ComplexDenoiserWrapper.md#deepinv.models.ComplexDenoiserWrapper)         | Complex-valued wrapper for a real-valued denoiser $\denoisername(\cdot, \sigma)$.                                                                            |
| [`deepinv.models.MinusOneOneDenoiserWrapper`](https://deepinv.org/api/stubs/deepinv.models.MinusOneOneDenoiserWrapper.md#deepinv.models.MinusOneOneDenoiserWrapper) | A wrapper for denoisers trained on $[x_{\mathrm{min}}, x_{\mathrm{max}}]$ images to be used with math:`[-1, 1]` images, i.e. on diffusion sampling iterates. |

## Deep Image Prior

**User Guide:** refer to [Deep Image Prior](https://deepinv.org/user_guide/reconstruction/adversarial.md#deep-image-prior) for more information.

| [`deepinv.models.DeepImagePrior`](https://deepinv.org/api/stubs/deepinv.models.DeepImagePrior.md#deepinv.models.DeepImagePrior)   | Deep Image Prior reconstruction.            |
|----------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| [`deepinv.models.ConvDecoder`](https://deepinv.org/api/stubs/deepinv.models.ConvDecoder.md#deepinv.models.ConvDecoder)         | Convolutional decoder network.              |
| [`deepinv.models.Poisson2Sparse`](https://deepinv.org/api/stubs/deepinv.models.Poisson2Sparse.md#deepinv.models.Poisson2Sparse)   | Poisson2Sparse model for Poisson denoising. |
| [`deepinv.models.ConvLista`](https://deepinv.org/api/stubs/deepinv.models.ConvLista.md#deepinv.models.ConvLista)             | Convolutional LISTA network.                |

## Adversarial Networks

**User Guide:** refer to [Adversarial Learning](https://deepinv.org/user_guide/training/loss.md#adversarial-losses) for more information.

| [`deepinv.models.PatchGANDiscriminator`](https://deepinv.org/api/stubs/deepinv.models.PatchGANDiscriminator.md#deepinv.models.PatchGANDiscriminator)   | PatchGAN Discriminator model.                                         |
|------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`deepinv.models.ESRGANDiscriminator`](https://deepinv.org/api/stubs/deepinv.models.ESRGANDiscriminator.md#deepinv.models.ESRGANDiscriminator)       | ESRGAN Discriminator.                                                 |
| [`deepinv.models.DCGANGenerator`](https://deepinv.org/api/stubs/deepinv.models.DCGANGenerator.md#deepinv.models.DCGANGenerator)                 | DCGAN Generator.                                                      |
| [`deepinv.models.DCGANDiscriminator`](https://deepinv.org/api/stubs/deepinv.models.DCGANDiscriminator.md#deepinv.models.DCGANDiscriminator)         | DCGAN Discriminator.                                                  |
| [`deepinv.models.CSGMGenerator`](https://deepinv.org/api/stubs/deepinv.models.CSGMGenerator.md#deepinv.models.CSGMGenerator)                   | Adapts a generator model backbone (e.g DCGAN) for CSGM or AmbientGAN. |

## Identification Models

**User Guide:** refer to [Blind Inverse Problems](https://deepinv.org/user_guide/reconstruction/blind.md#blind) for more information.

| [`deepinv.models.KernelIdentificationNetwork`](https://deepinv.org/api/stubs/deepinv.models.KernelIdentificationNetwork.md#deepinv.models.KernelIdentificationNetwork)     | Space varying blur kernel estimation network.    |
|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [`deepinv.models.WaveletNoiseEstimator`](https://deepinv.org/api/stubs/deepinv.models.WaveletNoiseEstimator.md#deepinv.models.WaveletNoiseEstimator)                 | Wavelet Gaussian noise level estimator.          |
| [`deepinv.models.PatchCovarianceNoiseEstimator`](https://deepinv.org/api/stubs/deepinv.models.PatchCovarianceNoiseEstimator.md#deepinv.models.PatchCovarianceNoiseEstimator) | Patch Covariance Gaussian noise level estimator. |
