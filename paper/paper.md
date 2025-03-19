---
title: 'DeepInverse: A Python package for solving imaging inverse problems with deep learning'
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

`deepinv` is an open-source pytorch-based library for solving imaging inverse problems with deep learning.
The library aims to cover most steps in modern imaging pipelines, from the definition of the forward sensing operator
to the training of unfolded reconstruction networks in a supervised or self-supervised way. 

# Statement of need

Deep neural networks have become ubiquitous in various imaging inverse problems, from computational photography to astronomical and medical imaging. Despite the ever-increasing research effort in the field, most learning-based algorithms are built from scratch, are hard to generalize beyond the specific problem they were designed to solve, and the results reported in papers are often hard to reproduce. `deepinv` aims to overcome these limitations, by providing a unified framework for defining imaging operators and solvers, which leverages popular pytorch deep learning library [@paszke2019pytorch], making most modules compatible with auto-differentiation.
The target audience of this library are both inverse problems researchers (experts in optimization, machine learning, etc.) and practitioners (biologists, physicists, etc.). `deepinv` has the following objectives: i) accelerate future research by enabling efficient testing and deployment
of new ideas, ii) enlarge the adoption of deep learning in inverse problems by lowering the entrance bar to new researchers 
in the field, and iii) enhance research reproducibility via a common definition of imaging operators and reconstruction
methods and a common framework for defining datasets for inverse problems.

While other python computational imaging libraries exist, to the best of our knowledge, `deepinv` is the only one with a focus learning-based methods.  SCICO [@balke2022scico], Pyxu [@simeoni2022pyxu] are python libraries whose main focus are variational optimization and/or plug-and-play reconstruction methods. These libraries do not provide specific tools for training reconstruction models such as trainers and custom losses, and do not cover non optimization-based solvers including diffusion methods, adversarial methods or unrolling networks. Moreover, `deepinv` provides a larger set of forward operators and generators. Another python library for computational imaging is ODL [@adler2018odl], which mostly focuses on computed tomography, and does not include a large variety of solvers. There are also multiple libraries focusing on specific inverse problems: ASTRA [@van2016astra] and the related pytomography [@polson2025pytomography] define advanced tomography operators, and PyLops [@ravasi2019pylops] provides a linear operator class and many built-in linear operators. Operator-specific libraries can be used together with deepinv as long as they are compatible with pytorch. 

![Schematic of the library.\label{fig:schematic}](figures/deepinv_schematic_.png)

# Inverse problems

Imaging inverse problems can be expressed as 
\begin{equation} \label{eq:solver}
y = N_{\sigma}(A_{\xi}(x))
\end{equation}
where $x\in\mathcal{X}$ is an image, $y\in\mathcal{Y}$ are the measurements, $A_{\xi}:\mathcal{X}\mapsto\mathcal{Y}$ is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
$N_{\sigma}:\mathcal{Y}\mapsto \mathcal{Y}$ is a mapping which characterizes the noise affecting the measurements parameterized by $\sigma$ (e.g., the noise level or gain). The forward operation is simply written in deepinv as `x_hat = physics(y, **params)` where `params` is a dictionary with optional forward operator parameters. Most forward operators in the library are matrix-free, scaling gracefully to large image sizes. The library provides high-level operator definitions which are associated with specific imaging applications (magnetic resonance imaging, computed tomography, radioastronomy, etc.), but also allows to perform operator algebra, like summing, concatenating or stacking operators. Moreover, the library provides multiple useful tools for handling linear operators, such as matrix-free linear solvers (conjugate gradient[@hestenes1952methods], MINRES[@paige1975solution], BiCGStab[@van1992bi] and LSQR[@paige1982lsqr]), and operator norm and condition number estimators. The table below summarizes the available forward operators:

| **Family**  | **Operators**  | **Generators**    
|-------------|----------------|---------------|
| Pixelwise | `Denoising`, `Inpainting`, `Demosaicing`, `Decolorize`  | `GaussianMaskGenerator`, `RandomMaskGenerator`, `EquispacedMaskGenerator` |
| Blur & Super-Resolution        | `Blur`, `BlurFFT`, `SpaceVaryingBlur`, `Downsampling` | `MotionBlurGenerator`, `DiffractionBlurGenerator`, `ProductConvolutionBlurGenerator`, `ConfocalBlurGenerator3D`, `DownsamplingGenerator3D`, `gaussian_blur`, `sinc_filter`, `bilinear_filter`, `bicubic_filter` |
| Magnetic Resonance Imaging (MRI) | `MRIMixin`, `MRI`, `MultiCoilMRI`, `DynamicMRI`, `SequentialMRI`, (All support 3D MRI) | `GaussianMaskGenerator`, `RandomMaskGenerator`, `EquispacedMaskGenerator` |
| Tomography | `Tomography` | |
| Remote Sensing & Multispectral | `Pansharpen`, `HyperSpectralUnmixing`, `CompressiveSpectralImaging` | |
| Compressive | `CompressedSensing`, `StructuredRandom`, `SinglePixelCamera` | |
| Radio Interferometric Imaging  | `RadioInterferometry` | |
| Single-Photon Lidar | `SinglePhotonLidar` | |
| Dehazing | `Haze` | |
| Phase Retrieval | `PhaseRetrieval`, `RandomPhaseRetrieval`, `StructuredRandomPhaseRetrieval`, `Ptychography`, `PtychographyLinearOperator` | `build_probe`, `generate_shifts` |

# Reconstruction methods

The library provides multiple solvers which depend on the forward operator and noise distribution:
\begin{equation} \label{eq:solver}
\hat{x} = \operatorname{R}_{\theta}(y, A_{\xi}, \sigma)
\end{equation}
where $\operatorname{R}_{\theta}$ is a reconstruction network/algorithm with trainable parameters $\theta$.
In deepinv code, a reconstructor is simply evaluated as `x_hat = model(y, physics)`.
The library covers a wide variety of existing approaches for bulding $\operatorname{R}_{\theta}$, ranging from classical variational optimization algorithms, to diffusion methods using plug-and-play denoisers. 

**Artifact Removal**: The simplest way architecture for solving inverse problems [@jin2017deep] is to backproject the measurements to the image domain and apply a denoiser (image-to-image) architecture such as a UNet. These architectures can be thus written as $\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) = \operatorname{D}_{\sigma}(A_{\xi}^{\top}y)$ where the adjoint can be replaced by any pseudoinverse of $A$.

**Variational Optimization**: This methods consist of solving an optimization problem [@chambolle2016introduction]
\begin{equation} \label{eq:var}
\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) = \operatorname{argmin}_{x} f(y,A_{\xi} x) + g(x)
\end{equation}
where $f:\mathcal{Y} \times \mathcal{Y} \mapsto \mathbb{R}_+$ is the data fidelity term which can incorporate knowledge about the noise parameters $\sigma$ and forward operator $A$, and $g:\mathcal{X}\mapsto\mathbb{R}_+$ is a regularization term that promotes plausible reconstructions. The library provides popular hand-crafted regularization functions, such as sparsity [@candes2008introduction] and total variation [@rudin1992nonlinear].

**Plug-and-Play**: Plug-and-play methods replace the proximal operator or gradient of the regularization term $g$ by a pretrained denoiser, i.e.,
 often using a deep denoiser [@kamilov2023plug]. We provide popular PnP strategies such as DPIR [@zhang2021plug].
 
**Diffusion and Langevin methods**:  As with Plug-and-Play methods, diffusion [@chung2022diffusion] [@kawar2022denoising] [@zhu2023denoising]  and Langevin methods [@laumont2022bayesian] incorporate prior information via a pretrained denoiser, however, they are associated to a stochastic differential equation or an ordinary differential equation, instead of the optimization of \eqref{eq:var}.

**Unfolded Networks and Deep Equilibrium**: Unfolded networks consist of fixing the number of optimization iterations of a variational or plug-and-play approach [@monga2021algorithm], and training the parameters of the resulting algorithm, including optimization parameters and possibly the regularization term parameters, including the deep denoiser in the case of PnP.

**Generative Adversarial Networks and Deep Image Prior**: Generative models exist in unconditional or conditional forms. Unconditional methods [@bora2017compressed] [@bora2018ambientgan] leverage a pretrained generator $G_{\theta}(z):\mathcal{Z}\mapsto \mathcal{X}$ where $z\in\mathcal{Z}$ is a latent code tol solve an inverse problem via
\begin{equation} \label{eq:var}
\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) = \operatorname{argmin}_{x} f(y,A_{\xi}G_{\theta}(z))
\end{equation}
The deep image prior [@ulyanov2018deeps] uses an untrained $\operatorname{G}_{\theta}$ leveraging the strong inductive bias of a specific autoencoder architecture. 

Conditional methods [@isola2017image] [@bendel2023gan] use adversarial training to learn a network $\operatorname{R}_{\theta}(y, z, A_{\xi}, \sigma)$ which provides a set of reconstructions by sampling different latent codes $z\in\mathcal{Z}$. 

**Foundation Models**: Foundation models are end-to-end architectures that incorporate knowledge of $A_{\xi}$ and $\sigma$ and are trained to reconstruct images across a wide variety of forward operators $A$ and noise distributions $N_{\sigma}$ [@terris2025ram]. Foundation models can be finetuned to unseen inverse problems using measurement data alone.

The table below summarizes all the categories of reconstructors considered in the library:


| **Family of Methods** | **Description** | **Training** | **Iterative** | **Sampling** |
|-----------------------|-----------------|-----------------------|---------------|--------------|
| Artifact Removal | Applies a neural network to a non-learned pseudo-inverse | Yes | No | No |   
| Variational Optimization | Solves an optimization problem with hand-crafted priors | No | Yes | No |    
| Plug-and-Play (PnP) | Leverages pretrained denoisers as priors within an optimization algorithm | No | Yes | No |
| Unfolded Networks | Constructs a trainable architecture by unrolling a PnP algorithm   | Yes | Only DEQ | No |  
| Diffusion & Langevin | Leverages pretrained denoisers within an SDE  | No | Yes | Yes |   
| Generative Adversarial Networks and Deep Image Prior | Uses a generator network to model the set of possible images | No | Yes | Depends |
| Foundation Models | Models trained for many imaging problems | Finetuning  | No   | No | 

# Training

The package provides losses for training $\operatorname{R}_{\theta}$ which are especially designed for inverse problems. 

**Supervised Losses**: Supervised losses using a dataset of ground-truth references $\{x_i\}_{i=1}^{N}$ or pairs $\{(x_i,y_i)\}_{i=1}^{N}$. If the forward model is known, measurements are typically generated directly during training.

| **Category**| **Loss**    | **Description** |
|-------------|-------------------|-----------------|
| End2End    | `SupLoss` | Requires paired data. 
| Adversarial    |  `SupAdversarialGeneratorLoss`, `SupAdversarialDiscriminatorLoss`|  Supervised adversarial loss.|

**Self-supervised Losses**: Self-supervised losses rely on measurement data only $\{y_i\}_{i=1}^{N}$ which can be roughly classified in splitting-based losses [@yaman2020self] [@krull2019noise2void] [@huang2021neighbor2neighbor] [@eldeniz2021phase2phase] [@liu2020rare], Stein's Unbiased Risk estimator and related losses [@pang2021recorrupted] [@tachella2025unsure] and nullspace losses [@chenequivariant2021] [@tachella2022unsupervised].

| **Category**      | **Loss**    | **Description** |
|-------------------|-------------|-----------------|
| Splitting  | `SplittingLoss`, `Neighbor2Neighbor`, `Phase2PhaseLoss`, `Artifact2ArtifactLoss` | Independent noise across measurements or pixels. Splitting across time. |
| SURE | `SureGaussianLoss`,  `SurePoissonLoss`, `SurePGLoss`,`R2RLoss` | Gaussian, Poisson, Poisson-Gaussian, or Gamma noise. |
| Nullspace losses | `EILoss`, `MOEILoss`, `MOEILoss` | Invariant distribution. Multiple operators |
| Adversarial | `UnsupAdversarialGeneratorLoss`, `UnsupAdversarialDiscriminatorLoss`, `UAIRGeneratorLoss` |  Unsupervised adversarial loss. Unsupervised reconstruction & adversarial loss. |
| Other | `TVLoss` | Total Variation regularization.|

**Network regularization losses**:  Network regularization losses which enforce some regularity condition on $\operatorname{R}_{\theta}$, generally having an upper bounded Lipschitz constant or similarly being firmly non-expansive [@pesquet2021learning].

|    **Loss**    | **Description** |
|----------------|-----------------|
| `JacobianSpectralNorm`, `FNEJacobianSpectralNorm` |  Promotes a firmly non-expansive network. |


# Acknowledgements

Julián Tachella acknowledges support by the ANR grant UNLIP (ANR-23-CE23-0013) and the CNRS PNRIA deepinverse project.
The authors would acknowledge the Jean-Zay high-performance computing center for providing computational resources.

# References