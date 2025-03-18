---
title: 'Deepinverse: A Python package for solving imaging inverse problems with deep learning'
tags:
  - Python
  - Pytorch
  - imaging inverse problems
  - computational imaging
authors:
  - name: Julián Tachella
    orcid: 0000-0003-3878-9142
    equal-contrib: true
    corresponding: true 
    affiliation: 1
  - name: Matthieu Terris
    equal-contrib: true
    affiliation: 2
  - name: Samuel Hurault
    equal-contrib: true 
    affiliation: 3
  - name: Andrew Wang
    equal-contrib: true 
    affiliation: 4
  - name: Dongdong Chen
    affiliation: 5
  - name: Hai Minh
    affiliation: 5
  - name: Maxime Song
    affiliation: 5
  - name: Leo Davy  
    affiliation: 1
  - name: Paul Escande  
    affiliation: 6
  - name: Theo Gnassou  
    affiliation: 6
  - name: Johannes Hertrich  
    affiliation: 6
  - name: Zhiyuan Hu  
    affiliation: 6
  - name: Tobias Liaudat  
    affiliation: 6
  - name: Nils Laurent  
    affiliation: 6
  - name: Brett Levac  
    affiliation: 6
  - name: Mathurin Massias  
    affiliation: 6
  - name: Thomas Moreau  
    affiliation: 6
  - name: Thibaut Mordzyk  
    affiliation: 6
  - name: Brayan Monroy  
    affiliation: 6
  - name: Antoine Regnier  
    affiliation: 5
  - name: Jérémy Scanvic  
    affiliation: 1
  - name: Florian Sarron  
    affiliation: 15
  - name: Victor Sechaud  
    affiliation: 1
  - name: Georg Schramm  
    affiliation: 4
  - name: Chao Tang  
    affiliation: 4
  - name: Jonathan Dong  
    affiliation: 4
  - name: Pierre Weiss  
    affiliation: 4  

affiliations:
  - name: CNRS, ENS de Lyon, Univ Lyon, Lyon, France
    index: 1
  - name: Inria Paris Saclay, Palaiseau, France
    index: 2
  - name: ENS Paris, Paris, France
    index: 3
  - name: University of Edinburgh, Edinburgh, UK
    index: 3
  - name: Heriot-Watt University, Edinburgh, UK
    index: 3
date: 15 March 2025
bibliography: paper.bib

---

# Summary

`deepinv` is an open-source PyTorch-based library for solving imaging inverse problems with deep learning.
The library aims to cover most steps in modern imaging pipelines, from the definition of the forward sensing operator
to the training of unfolded reconstruction networks in a supervised or self-supervised way.

# Statement of need

Deep neural networks have become ubiquitous in various imaging inverse problems, from computational photography
to astronomical and medical imaging. Despite the ever-increasing research effort in the field, most learning-based algorithms
are built from scratch, are hard to generalize beyond the specific problem they were designed to solve, and the results 
reported in papers are often hard to reproduce. In order to tackle these pitfalls, an international group of research scientists has worked intensely in the last year to build 
`deepinv` is an open-source PyTorch library [@paszke2019pytorch] for solving imaging inverse problems with deep learning, whose first stable
version has been very recently released. DeepInverse aims to cover most of the steps in modern imaging pipelines,
from the definition of the forward sensing operator to the training of unfolded reconstruction networks in a supervised or self-supervised way. 
The main goal of this library is to become the standard open-source tool for researchers (experts in optimization,
machine learning, etc.) and practitioners (biologists, physicists, etc.). 
The `deepinv` has the following objectives: i) accelerate future research by enabling efficient testing and deployment
of new ideas, ii) enlarge the adoption of deep learning in inverse problems by lowering the entrance bar to new researchers 
in the field, and iii) enhance research reproducibility via a common definition of imaging operators and reconstruction
methods and a common framework for defining datasets for inverse problems.


![Schematic of the library.\label{fig:schematic}](figures/deepinv_schematic_.png)


# Inverse problems

Imaging inverse problems can be expressed as 
\begin{equation} \label{eq:solver}
y = N(A(x))
\end{equation}
where $x\in\mathcal{X}$ is an image, $y\in\mathcal{Y}$ are the measurements, $A:\mathcal{X}\mapsto\mathcal{Y}$ is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
$N:\mathcal{Y}\mapsto \mathcal{Y}$ is a mapping which characterizes the noise affecting the measurements.
The forward operation is simply written in deepinv as `x_hat = physics(y)`.


| **Family**  | **Operators**  | **Generators**    
|-------------|----------------|---------------|
| Pixelwise         | `Denoising`, `Inpainting`, `Demosaicing`, `Decolorize`                  | `BernoulliSplittingMaskGenerator`, `GaussianSplittingMaskGenerator`, `Phase2PhaseSplittingMaskGenerator`, `Artifact2ArtifactSplittingMaskGenerator`                                                     |
| Blur & Super-Resolution        | `Blur`, `BlurFFT`, `SpaceVaryingBlur`, `Downsampling`                   | `MotionBlurGenerator`, `DiffractionBlurGenerator`, `ProductConvolutionBlurGenerator`, `ConfocalBlurGenerator3D`, `DownsamplingGenerator3D`, `gaussian_blur`, `sinc_filter`, `bilinear_filter`, `bicubic_filter`               |
| Magnetic Resonance Imaging (MRI) | `MRIMixin`, `MRI`, `MultiCoilMRI`, `DynamicMRI`, `SequentialMRI`, (All support 3D MRI) | `GaussianMaskGenerator`, `RandomMaskGenerator`, `EquispacedMaskGenerator`,                                                                                         |
| Tomography                     | `Tomography`                                                                  |                                                                                                                                                                                                               |
| Remote Sensing & Multispectral | `Pansharpen`, `HyperSpectralUnmixing`, `CompressiveSpectralImaging`       |                                                                                                                                                                                                               |
| Compressive                    | `CompressedSensing`, `StructuredRandom`, `SinglePixelCamera`              |                                                                                                                                                                                                               |
| Radio Interferometric Imaging  | `RadioInterferometry`                                                         |                                                                                                                                                                                                               |
| Single-Photon Lidar            | `SinglePhotonLidar`                                                           |                                                                                                                                                                                                               |
| Dehazing                       | `Haze`                                                                        |                                                                                                                                                                                                               |
| Phase Retrieval                | `PhaseRetrieval`, `RandomPhaseRetrieval`, `StructuredRandomPhaseRetrieval`, `Ptychography`, `PtychographyLinearOperator` | `build_probe`, `generate_shifts`                                                                                                                                                                             |

       
# Reconstruction methods

The library provides multiple solvers which depend on the acquisition physics:
\begin{equation} \label{eq:solver}
\hat{x} = R_{\theta}(y, A, N)
\end{equation}
where $R_{\theta}$ is a reconstruction network/algorithm with trainable parameters $\theta$ which can depends on the forward operator and noise distribution.
In deepinv code, a reconstructor is evaluated as `x_hat = solver(y, A)`.


| **Family of Methods**                     | **Description**                                                                 | **Requires Training** | **Iterative** | **Sampling** | **References** |
|-------------------------------------------|--------------------------------------------------------------------------------|-----------------------|---------------|--------------|----------------|
| Artifact Removal    | Applies a neural network to a non-learned pseudo-inverse            | Yes                   | No            | No           |    [@jin2017deep]            |
| Plug-and-Play (PnP)                       | Leverages pretrained denoisers as priors within an optimization algorithm        | No                    | Yes           | No           |          |
| Unfolded Networks                         | Constructs a trainable architecture by unrolling a PnP algorithm                 | Yes                   | Only DEQ      | No           |                |
| Diffusion                                 | Leverages pretrained denoisers within an ODE/SDE                                 | No                    | Yes           | Yes          |                |
| Non-learned Priors                        | Solves an optimization problem with hand-crafted priors                          | No                    | Yes           | No           |                |
| Markov Chain Monte Carlo (MCMC)           | Leverages pretrained denoisers as priors within an optimization algorithm        | No                    | Yes           | Yes          |                |
| Generative Adversarial Networks and Deep Image Prior | Uses a generator network to model the set of possible images                     | No                    | Yes           | Depends      |                |
| Specific Network Architectures            | Off-the-shelf architectures for specific inverse problems                        | Yes                   | No            | No           |                |
| Foundation Models            | Models trained for many imaging problems                        | Finetuning                   | No            | No           |                |



# Training

The package contains losses for training $R_{\theta}$ which are especially designed for inverse problems.
The losses can be roughly separated in 3 categories: i) Supervised losses using a dataset of ground-truth references $\{x_i\}_{i=1}^{N}$, ii) self-supervised losses using measurement data only $\{y_i\}_{i=1}^{N}$  and iii) network regularization losses which enforce some regularity condition on $R_{\theta}$.

| **Category**      | **Loss**    | **Assumptions** |
|-------------------|-------------|-----------------|
|| **Supervised Learning**  || 
| End2End    | `SupLoss` | Requires paired data. 
| Adversarial    |  `SupAdversarialGeneratorLoss`, `SupAdversarialDiscriminatorLoss`                  |  Supervised adversarial loss.                             |
|| **Self-Supervised Learning** ||
| Splitting  | `SplittingLoss`, `Neighbor2Neighbor`, `Phase2PhaseLoss`, `Artifact2ArtifactLoss` | Independent noise across measurements or pixels. Splitting across time.       |
|   SURE and Related Losses     | `SureGaussianLoss`,  `SurePoissonLoss`, `SurePGLoss`,`R2RLoss`                                           | Gaussian, Poisson, Poisson-Gaussian, or Gamma noise.                          |
|     Nullspace losses      | `EILoss`, `MOEILoss`, `MOEILoss`          | Invariant distribution. Multiple operators |
|   Adversarial       | `UnsupAdversarialGeneratorLoss`, `UnsupAdversarialDiscriminatorLoss`, `UAIRGeneratorLoss` |  Unsupervised adversarial loss. Unsupervised reconstruction & adversarial loss. |
|   Other       | `TVLoss`| Total Variation regularization.|
|| **Network Regularization** ||
| | `JacobianSpectralNorm`, `FNEJacobianSpectralNorm`                                            | Controls the spectral norm of the Jacobian matrix. Promotes a firmly non-expansive network. |


# Acknowledgements

Julián Tachella acknowledges support by the ANR grant UNLIP (ANR-23-CE23-0013) and the CNRS PNRIA deepinverse project.
The authors would acknowledge the Jean-Zay high-performance computing center for providing computational resources.

# References