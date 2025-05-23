---
title: 'DeepInverse: A Python package for solving imaging inverse problems with deep learning'
tags:
  - Python
  - PyTorch
  - imaging inverse problems
  - computational imaging
  - deep learning
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
    orcid: 0000-0003-0838-7986
    equal-contrib: true 
    affiliation: 4
  - name: Dongdong Chen
    affiliation: 5
  - name: Minh-Hai Nguyen
    affiliation: 8
  - name: Maxime Song
    affiliation: 14
  - name: Thomas Davies
    affiliation: 5
  - name: Leo Davy  
    affiliation: 1
  - name: Jonathan Dong  
    affiliation: 7
  - name: Paul Escande  
    affiliation: 9
  - name: Johannes Hertrich  
    affiliation: 15
  - name: Zhiyuan Hu  
    affiliation: 7
  - name: Tobías I. Liaudat  
    affiliation: 13
  - name: Nils Laurent  
    affiliation: 6
  - name: Brett Levac  
    affiliation: 12
  - name: Mathurin Massias  
    affiliation: 10
  - name: Thomas Moreau  
    affiliation: 2
  - name: Thibaut Modrzyk  
    affiliation: 11
  - name: Brayan Monroy  
    affiliation: 6
  - name: Sebastian Neumayer
    affiliation: 16
  - name: Jérémy Scanvic  
    affiliation: 1
  - name: Florian Sarron  
    affiliation: 8
  - name: Victor Sechaud  
    affiliation: 1
  - name: Georg Schramm  
    affiliation: 17
    orcid: 0000-0002-2251-3195
  - name: Chao Tang  
    affiliation: "4, 5"
  - name: Romain Vo
    affiliation: 1
  - name: Pierre Weiss  
    affiliation: 8  

affiliations:
  - name: CNRS, ENS de Lyon, Univ Lyon, Lyon, France
    index: 1
  - name: Université Paris-Saclay, Inria, CEA, Palaiseau, France
    index: 2
  - name: CNRS, ENS Paris, PSL
    index: 3
  - name: University of Edinburgh, Edinburgh, UK
    index: 4
  - name: Heriot-Watt University, Edinburgh, UK
    index: 5
  - name: Universidad Industrial de Santander, Bucaramanga, Colombia
    index: 6
  - name: EPFL, Lausanne, Switzerland
    index: 7
  - name: IRIT, CBI, CNRS, Université de Toulouse, Toulouse, France. 
    index: 8
  - name: IMT, CNRS, Université de Toulouse, Toulouse, France
    index: 9
  - name: Inria, ENS de Lyon, Univ Lyon, Lyon, France
    index: 10
  - name: INSA de Lyon, Lyon, France
    index: 11
  - name: University of Texas at Austin, Austin, USA
    index: 12
  - name: IRFU, CEA, Université Paris-Saclay, Gif-sur-Yvette, France
    index: 13
  - name: CNRS UAR 851, Université Paris-Saclay Orsay, France
    index: 14
  - name: Université Paris Dauphine - PSL, Paris, France
    index: 15
  - name: Chemnitz University of Technology, Chemnitz, Germany
    index: 16
  - name: Department of Imaging and Pathology, KU Leuven, Leuven, Belgium
    index: 17

date: 15 May 2025
bibliography: paper.bib

---

# Summary

DeepInverse is an open-source PyTorch-based library for imaging inverse problems.
The library covers all crucial steps in image reconstruction from the efficient implementation of forward operators (optics, MRI, tomography,...), 
to the definition and resolution of variational problems and the design and training of advanced neural network architectures. 

# Statement of Need

Deep neural networks have become ubiquitous in various imaging inverse problems, from computational photography to astronomical and medical imaging. Despite the ever-increasing research effort in the field, most learning-based algorithms are built from scratch, are hard to generalize beyond their specific training setting, and the reported results are often hard to reproduce. `deepinv` overcomes these limitations by providing a unified framework for defining imaging operators and solvers. It leverages the popular PyTorch deep learning library [@paszke2019pytorch], making most modules compatible with auto-differentiation.
The target audience of this library are both researchers in inverse problems (experts in optimization, machine learning, etc.) and practitioners (biologists, physicists, etc.). `deepinv` has the following objectives:

1. **Accelerate research** by enabling efficient testing, deployment and transfer
of new ideas across imaging domains;
2. Enlarge the **adoption of deep learning in inverse problems** by lowering the entrance bar to new researchers and practitioners;
3. Enhance **research reproducibility** via a common framework for imaging operators, reconstruction
methods, datasets, and metrics for inverse problems.

While other Python computational imaging libraries exist, to the best of our knowledge, `deepinv` is the only one with a strong focus on learning-based methods which provides a larger set of realistic imaging operators.
SCICO [@balke2022scico] and Pyxu [@simeoni2022pyxu] are Python libraries whose main focus are variational optimization and/or plug-and-play reconstruction methods. These libraries do not provide specific tools for training reconstruction models such as trainers and custom loss functions, and do not cover non optimization-based solvers including diffusion methods, adversarial methods or unrolling networks.
CUQIpy [@riis2024cuqipy] is a library focusing on Bayesian uncertainty quantification methods for inverse problems.
TIGRE [@biguri2025tigre], ODL [@adler2018odl] and CIL [@jorgensen2021core] mostly focus on computed tomography and do not cover deep learning pipelines for inverse solvers. 
There are also multiple libraries focusing on specific inverse problems: ASTRA [@van2016astra] and the related pytomography [@polson2025pytomography] define advanced tomography operators, sigpy [@ong2019sigpy] provides magnetic resonance imaging (MRI) operators without deep learning, and PyLops [@ravasi2019pylops] provides a linear operator class and many built-in linear operators.
These operator-specific libraries can be used together with `deepinv` as long as they are compatible with PyTorch (for example, we provide a wrapper for ASTRA). Advanced libraries for inverse problems also exist in other programming languages such as MATLAB, including GlobalBioIm [@soubies2019pocket] or IR Tools [@gazzola2019ir], but they are restricted to handcrafted reconstruction methods without automatic differentiation. 


![Schematic of the library.\label{fig:schematic}](../docs/source/figures/deepinv_schematic.png)

# Inverse Problems

Imaging inverse problems can be expressed as 
\begin{equation} \label{eq:solver}
y = N_{\sigma}(A_{\xi}(x))
\end{equation}
where $x\in\mathcal{X}$ is an image, $y\in\mathcal{Y}$ are the measurements, $A_{\xi}\colon\mathcal{X}\mapsto\mathcal{Y}$ is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
$N_{\sigma}\colon\mathcal{Y}\mapsto \mathcal{Y}$ is a mapping that characterizes the noise affecting the 
measurements parameterized by $\sigma$ (e.g. the noise level and/or gain). The forward operation is simply written in `deepinv` as `x_hat = physics(y, **params)` where `params` is a dictionary with
optional parameters $\xi$. This framework unifies the wide variety of forward operators across various domains. Most forward operators in `deepinv` are matrix-free, scaling gracefully to large image sizes.
The library provides high-level operator definitions which are associated with specific imaging applications 
(MRI, computed tomography, radioastronomy, etc.), and allows users to perform operator algebra,
like summing, concatenating or stacking operators. `deepinv` comes with multiple useful tools for handling
linear operators, such as adjoint, pseudoinverses and proximal operators (leveraging the singular value decomposition when possible), matrix-free linear solvers [@hestenes1952methods] [@paige1975solution] [@van1992bi],
and operator norm and condition number estimators [@paige1982lsqr]. Many common noise distributions are included in the library
such as Gaussian, Poisson, mixed Poisson-Gaussian, uniform and gamma noise. The table below summarizes the available forward operators
at the time of writing, which are constantly being expanded and improved upon by the community:

| **Family**                     | **Operators** $A$                                                   | **Generators**   $\xi$                                     |
|--------------------------------|---------------------------------------------------------------------|------------------------------------------------------------|
| Pixelwise                      | Denoising, inpainting, demosaicing, decolorize                      | Mask generators, noise level generators                                        |
|                       |                                                                     |                          |
| Blur & Super-Resolution        | Blur, space-varying blur, downsampling                              | Motion, Gaussian, diffraction, product-convolution, and confocal blurs. Sinc, bilinear and bicubic antialias filters. |
|                       |                                                                     |                          |
| Magnetic Resonance Imaging (MRI) | Single and multi-coil, dynamic and sequential (All support 3D MRI)  | Gaussian, random and equispaced masks |
|                       |                                                                     |                          |
| Tomography                     | 2D parallel beam, 2D fanbeam, 3D parallel beam and 3D conebeam                      |                                                                                |
|                       |                                                                     |                          |
| Remote Sensing & Multispectral | Pansharpening, hyperspectral unmixing, compressive spectral imaging |                                                                                |
|                       |                                                                     |                          |
| Compressive                    | Compressed sensing and single-pixel camera                          |                                                                                |
|                       |                                                                     |                          |
| Radio Interferometric Imaging  | Monochromatic intensity imaging with narrow field-of-view           |                                                                                |
|                       |                                                                     |                          |
| Single-photon Lidar            | TCSPC lidar                                                         |                                                                                |
|                       |                                                                     |                          |
| Dehazing                       | Parametric haze model                                               |                                                                                |
|                       |                                                                     |                          |
| Phase Retrieval                | Random operators and ptychography                                   | Probe generation |


### Operator Parameterization

Most physics operators are parameterized by a vector $\xi$, which has a different meaning depending on the context.
For instance, it can represent the projection angles in tomography, the blur kernel in image deblurring, the acceleration masks in MRI, etc.
Integrating this parameter allows for advanced computational imaging problems, including calibration of the system (measuring $\xi$ from $y$),
blind inverse problems (recovering $\xi$ and $x$ from $y$) [@debarnot2024deep] [@chung2023parallel], co-design (optimizing $\xi$ and possibly 
the reconstruction algorithm jointly) [@lazarus2019sparkling] [@nehme2020deepstorm3d], robust neural network training [@gossard2024training]
[@terris2023meta] [@terris2025ram]. To the best of our knowledge, this feature is distinctive and becoming essential in recent advances
in image reconstruction.


# Reconstruction Methods

The library provides multiple solvers which depend on the forward operator $A_{\xi}$ and the noise distribution via $\sigma$. Our framework unifies the wide variety of solvers that are commonly used in the current literature:
\begin{equation} \label{eq:solver}
\hat{x} = \operatorname{R}_{\theta}(y, A_{\xi}, \sigma)
\end{equation}
where $\operatorname{R}_{\theta}$ is a reconstruction network/algorithm with (optional) trainable parameters $\theta$ (see *"Training"*).
In `deepinv` code, a reconstructor is simply evaluated as `x_hat = model(y, physics)`.
The library covers a wide variety of approaches for building $\operatorname{R}_{\theta}$, which can be roughly divided into
optimization-based methods, sampling-based methods, and non-iterative methods. 

### Optimization-Based Methods

These methods consist of solving an optimization problem [@chambolle2016introduction]
\begin{equation} \label{eq:var}
\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) \in \operatorname{argmin}_{x} f_{\sigma}(y,A_{\xi}(x)) + g(x)
\end{equation}
where $f_{\sigma}\colon\mathcal{Y} \times \mathcal{Y} \mapsto \mathbb{R}$ is the data fidelity term, which can incorporate knowledge about the noise parameters $\sigma$, and $g\colon\mathcal{X}\mapsto\mathbb{R}$ is a regularizer that promotes plausible reconstructions.
The `optim` module includes classical fidelity terms (e.g., $\ell_1$, $\ell_2$, Poisson log-likelihood) and offers a wide range of regularization priors:

**Explicit Priors**: The library implements several traditional regularizers, such as sparsity [@candes2008introduction], total variation [@rudin1992nonlinear], wavelets [@stephane1999wavelet], patch-based likelihoods [@zoran2011learning], and mixed-norm regularizations [@kowalski2009sparse].
We also provide learned regularizers such as EPLL [@zoran2011epll] and PatchNR [@altekruger2023patchnr].

**Denoising Plug-and-Play Priors**: Plug-and-Play (PnP) methods replace the proximal operator [@venkatakrishnan2013plug] or the gradient [@romano2017little] of the regularizer $g$ with a pretrained denoiser, often based on deep learning. The library provides access to widely used pretrained denoisers $\operatorname{D}_{\sigma}$, many of them trained on multiple noise levels $\sigma$, including DnCNN [@zhang2017beyond], DRUNet [@zhang2021plug], and recent diffusion-based denoisers such as DDPM [@ho2020denoising] and NCSN [@song2020score].

The `optim` module also includes solvers for the minimization problem in \eqref{eq:var} using:

**Optimization Algorithms**: The library contains several classical algorithms [@dossal2024optimizationorderalgorithms] for minimizing the sum of two functions, including proximal gradient descent, FISTA, ADMM, Douglas-Rachford Splitting, and primal-dual methods.

**Unfolded Networks and Deep Equilibrium Models**: Unfolded networks [@gregor2010learning] are obtained by unrolling a fixed number of iterations of an optimization algorithm and training the parameters end-to-end, including both optimization hyperparameters and deep regularization priors. Standard unfolded methods train via backpropagation through the optimization algorithm, while deep equilibrium methods [@bai2019deep] implicitly differentiate the fixed point of the algorithm.



### Sampling-Based Methods
Reconstruction methods can also be defined via ordinary or stochastic differential equations, generally as a Markov chain defined by
\begin{equation}
x_{t+1} \sim p(x_{t+1}|x_t, y, \operatorname{R}_{\theta}, A_{\xi}, \sigma) \text{ for } t=0,\dots,T-1
\end{equation} 
such that $x_{T}$ is approximately sampled from the posterior $p(x|y)$, and $\operatorname{R}_{\theta}$ is a (potentially learned) denoiser.
Sampling multiple plausible reconstructions enables uncertainty estimates by computing statistics across the samples.
The sampling module provides implementations of the following methods:

**Diffusion Models**: In a similar fashion to PnP methods, diffusion models [@chung2022diffusion] [@kawar2022denoising] [@zhu2023denoising] incorporate prior information via a pretrained denoiser, however, they are linked to a stochastic differential equation (SDE) or an ordinary differential equation (ODE), instead of the optimization of \eqref{eq:var}.

**Langevin-Type Algorithms**: The library provides popular high-dimensional Markov Chain Monte Carlo (MCMC) methods such as Unadjusted Langevin Algorithm and some of its variants [@laumont2022bayesian] [@pereyra2020skrock], which
define a Markov chain with stationary distribution close to the posterior distribution $p(x|y)$.

### Non-Iterative Methods
Non-iterative methods are part of the `models` module, and include artifact removal, unconditional and conditional generative networks, and foundation models.

**Artifact Removal**: The simplest way of incorporating the forward operator into a network architecture is to backproject the measurements to the image domain and apply a denoiser (image-to-image) architecture $\operatorname{D}_{\sigma}$ such as a UNet [@jin2017deep].
These architectures can be thus written as

\begin{equation}
\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) = \operatorname{D}_{\sigma}(A_{\xi}^{\top}y),
\end{equation}
where the backprojection can be replaced by any pseudoinverse of $A_{\xi}$. 

**Unconditional Generative Networks**: Generative models exist in unconditional or conditional forms. Unconditional methods [@bora2017compressed] [@bora2018ambientgan] leverage
a pretrained generator $\operatorname{G}_{\theta}(z)\colon\mathcal{Z}\mapsto \mathcal{X}$ where $z\in\mathcal{Z}$ is a latent code to solve an inverse problem via
\begin{equation} \label{eq:cond}
\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) = \operatorname{G}_{\theta}(\hat{z}) \text{ with } \hat{z} \in \operatorname{argmin}_{z} f_{\sigma}(y,A_{\xi}(\operatorname{G}_{\theta}(z)))
\end{equation}
The deep image prior [@ulyanov2018deep] is a specific case of unconditional models which uses an untrained generator $\operatorname{G}_{\theta}$, leveraging the inductive bias of a specific autoencoder architecture. 

**Conditional Generative Networks**: Conditional generative adversarial networks [@isola2017image] [@bendel2023gan] use adversarial training to learn a network $\operatorname{R}_{\theta}(y, z, A_{\xi}, \sigma)$ which provides a set of reconstructions by sampling different latent codes $z\in\mathcal{Z}$. 

**Foundation Models**: Foundation models are end-to-end architectures that incorporate knowledge of $A_{\xi}$ and $\sigma$ and are trained to reconstruct images across a wide variety of forward operators $A_{\xi}$ and noise distributions $N_{\sigma}$ [@terris2025ram].
These models often obtain good performance in new tasks without retraining, and can also be finetuned to specific inverse problems or datasets using measurement data alone.

The table below summarizes the reconstruction methods considered in the library:

| **Family of Methods** | **Description**                                                 | **Training** | **Iterative** | **Sampling** |
|-----------------------|-----------------------------------------------------------------|--------------|---------------|--------------|
| Artifact Removal | Applies network to a pseudoinverse                              | Yes          | No | No |   
|   |                                                                 |              |  |  |   
| Variational Optimization | Solves optimization problem with hand-crafted/learned priors    | Hyperparams  | Yes | No |    
|   |                                                                 |              |  |  |   
| Plug-and-Play (PnP) | Leverages pretrained denoisers as priors within an optimization | Hyperparams           | Yes | No |
|   |                                                                 |              |  |  |   
| Unfolded Networks | Trainable architecture by unrolling an optimization             | Yes          | Only DEQ | No |  
|   |                                                                 |              |  |  |   
| Diffusion & Langevin | Leverages pretrained denoisers within an ODE/SDE                | Hyperparams           | Yes | Yes |   
|   |                                                                 |              |  |  |   
| Generative Adversarial Networks | Model plausible images via generator                            | Yes          | Yes | Depends |
|   |                                                                 |              |  |  |   
| Foundation Models | Models trained for many imaging problems                        | Finetune     | No   | No |

# Training

Training can be performed using the `Trainer` class, which is a high-level interface for training reconstruction networks. `Trainer` handles data ingestion, the training loop, logging, and checkpointing. 

## Losses
The library also provides the `loss` module with losses for training $\operatorname{R}_{\theta}$ which are especially designed for inverse problems, acting as a framework that unifies loss functions that are widely used in various inverse problems across domains. Loss functions are defined as

\begin{equation} \label{eq:loss}
l = \mathcal{L}\left(\hat{x}, x, y, A_{\xi}, \operatorname{R}_{\theta}\right)
\end{equation}

with $\hat{x}=R_{\theta}(y,A_{\xi},\sigma)$ being the network prediction, 
and written in `deepinv` as `l = loss(x_hat, x, y, physics, model)`, where
some inputs might be optional (e.g., $x$ is not needed for self-supervised losses).

**Supervised Losses**: Supervised learning can be done using a dataset of ground-truth and measurements pairs $\{(x_i,y_i)\}_{i=1}^{N}$ by applying a metric to compute the distance between $x$ and $\hat{x}$.
If the forward model is known, measurements are typically generated directly during training from a dataset of ground-truth references $\{x_i\}_{i=1}^{N}$ .

**Self-Supervised Losses**: Self-supervised losses rely on measurement data only $\{y_i\}_{i=1}^{N}$. We provide implementations of state-of-the-art losses from the literature [@wang2025benchmarking]. These can be roughly classified in three classes:

The first class consists of splitting losses [@batson2019noise2self], with operator-specific solutions for denoising [@krull2019noise2void] [@huang2021neighbor2neighbor] and MRI [@yaman2020self] [@eldeniz2021phase2phase] [@liu2020rare].
The second class are Stein's Unbiased Risk Estimate (SURE) and related losses: we provide variants of SURE for Gaussian, Poisson and Poisson-Gaussian noise respectively, which can also be used without knowledge of the noise levels [@tachella2025unsure]. The library includes the closely related Recorrupted2Recorrupted [@pang2021recorrupted] which handles Gaussian, Poisson and Gamma noise distributions [@monroy2025gr2r].
The third class corresponds to nullspace losses, exploiting invariance of the image distribution to transformations [@chen2021equivariant] or access to multiple forward operators [@tachella2022unsupervised]. 

The library provides image transforms $\tilde{x}=T_g x$ in the `transform` module where $g\in G$ parametrizes a transformation group, including a group-theoretic model for geometric transforms covering linear (Euclidean) and non-linear (projective, diffeomorphism) transforms [@wang2024perspective]. These can be used for data augmentation, equivariant imaging [@chen2021equivariant] and equivariant networks.

**Network Regularization Losses**:  Network regularization losses which try to enforce some regularity condition on $\operatorname{R}_{\theta}$, such as having an upper bounded Lipschitz constant or similarly being firmly non-expansive [@pesquet2021learning].

**Adversarial Losses**:  Adversarial losses are used to train conditional or unconditional generative networks. The library provides 
both supervised [@bora2017compressed] and self-supervised (i.e. no ground-truth) [@bora2018ambientgan] adversarial losses.
Due to the additional complexity of training adversarial networks, the library provides a specific `AdversarialTrainer` sub-class.

## Datasets
The library provides a common framework for defining and simulating datasets for image reconstruction. Datasets return ground-truth and measurements pairs $\{(x_i,y_i)\}_{i=1}^{N}$, and may also return physics parameters $\xi_i$. Given a dataset of reference images $\{x_i\}_{i=1}^{N}$, the library can be used to generate and save a simulated paired dataset to encourage reproducibility. The library also provides interfaces to some popular datasets to facilitate research in specific application domains: 

- Natural images: Div2K [@agustsson2017ntire], Urban100 [@lim2017enhanced], Set14 [@zeyde2012single], CBSD68 [@martin2001database], Flickr2K [@lim2017enhanced], LSDIR [@li2023lsdir];
- MRI scans: FastMRI [@zbontar2018fastmri], CMRxRecon [@wang2024cmrxrecon];
- Computed tomography scans: LIDC-IDRI [@armato2011lung];
- Fluorescence microscopy images: FMD [@zhang2019poisson]; 
- Real motion blur images: Kohler [@kohler2012recording];
- Multispectral satellite images: NBU [@meng2021pansharpening].

# Evaluation
Reconstruction methods can be evaluated on datasets with the method `Trainer.test` using metrics defined in our framework.
These are written in `deepinv` as `m = metric(x_hat, x)` in the case of full-reference metrics, or as `m = metric(x_hat)` for no-reference metrics, and provide common functionality such as input normalization and complex magnitude.

Following the distortion-perception trade-off in image reconstruction problems [@blau2018perception], 
the library provides common popular distortion metrics such as PSNR, SSIM [@wang2004image], and LPIPS [@zhang2018unreasonable], 
as well as no-reference perceptual metrics such as NIQE [@mittal2012making] and QNR [@yeganeh2012objective], which can be used to evaluate the quality of the reconstructions.

# Philosophy

### Coding Practices

`deepinv` is coded in modern Python following a test-driven development philosophy.
The code is unit-, integration- and performance-tested using `pytest` and verified using `codecov` (automatically via GitHub actions),
and is compliant with PEP8 using `black`.  To encourage reproducibility, the library passes random number generators
for all random generation functionality. Architecturally, `deepinv` is implemented using an object-oriented framework
where base classes provide abstract functionality and interfaces (such as `Physics` or `Metric`),
sub-classes provide specific implementations or special cases (such as `LinearPhysics`) along with methods inherited
from base classes (such as the operator pseudoinverse), and mixins provide specialized methods. This framework
reduces code duplication and makes it easy for researchers, engineers and practitioners to implement new or specialized
functionality while inheriting existing methods.

### Documentation

The library provides a **user-guide**, quickstart and in-depth **examples** for all levels of user, and individual API documentation
for classes. The documentation is built using Sphinx. We use Sphinx-Gallery [@najera2023sphinxgallery] for generating jupyter notebook demos, which 
are automatically tested and included in the documentation. The user-guide provides an overview of the library and is intended as a starting point for new users, whereas the API lists all classes and functions, being
intended for advanced users. The user-guide also serves as a computational imaging tutorial, providing an overview of
common imaging operators and reconstruction methods.
The documentation of most classes includes a usage example which is automatically tested using `doctest`, and 
a detailed mathematical description using latex with shared math symbols and notation across the whole documentation.

# Perspectives

DeepInverse is a dynamic and evolving project and this paper is merely a snapshot of ongoing progress (release v0.3.0). The community is continuously contributing more methods, such as more realistic physics operators and more advanced training techniques, which reflect the state-of-the-art in imaging with deep learning, aiming to address the needs and interests of researchers and practitioners alike.

# Acknowledgements

J. Tachella acknowledges support by the French National Research Agency (Agence Nationale de la Recherche) grant UNLIP (ANR-23-CE23-0013) and the CNRS PNRIA deepinverse project.
M. Terris acknowledges support by the BrAIN grant (ANR-20-CHIA-0016).
F. Sarron, P. Weiss, M.H. Nguyen were supported by the ANR Micro-Blind ANR-21-CE48-0008.
Thomas Moreau was supported from a national grant attributed to the ExaDoST project of the NumPEx PEPR program, under the reference ANR-22-EXNU-0004.
J. Hertrich is supported by the German Research Foundation (DFG) with project number 530824055.
Z. Hu acknowledges funding from the Swiss National Science Foundation (Grant PZ00P2_216211).
S. Neumayer acknowledges funding from the German Research Foundation (DFG) with project number 543939932.
The authors acknowledge the Jean-Zay high-performance computing center, using HPC resources from GENCI-IDRIS (Grants 2021-AD011012210, 2024-AD011015191). 

# References
