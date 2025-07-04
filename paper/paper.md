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
    affiliation: "4, 5"
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
  - name: Romain Vo
    affiliation: 1
  - name: Pierre Weiss  
    affiliation: 8  

affiliations:
  - name: CNRS, ENS de Lyon, Univ Lyon, Lyon, France
    index: 1
  - name: Université Paris-Saclay, Inria, CEA, Palaiseau, France
    index: 2
  - name: CNRS, ENS Paris, PSL, Paris, France
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

[DeepInverse](https://deepinv.github.io/) is an open-source PyTorch-based library for imaging inverse problems. DeepInverse implements all crucial steps in image reconstruction from imaging forward operators across a wide set of domains (medical imaging, astronomical imaging, remote sensing, computational photography, compressed sensing, and more)
to reconstruction algorithms and advanced neural network training. 

# Statement of Need

Deep neural networks have become ubiquitous in various imaging inverse problems. Despite the ever-increasing research effort in the field, most learning-based algorithms are built from scratch, are hard to generalize beyond their specific training setting, and the reported results are often hard to reproduce. `deepinv` overcomes these limitations by providing a modular unified framework, leveraging the popular PyTorch deep learning library [@paszke2019pytorch], making most modules compatible with auto-differentiation.
The target audience of this library is both researchers in inverse problems (experts in optimization, machine learning etc.), practitioners (biologists, physicists etc.) and imaging software engineers. `deepinv` has the following objectives:

1. **Accelerate research** by enabling efficient testing, deployment and transfer
of new ideas across imaging domains;
2. Enlarge the **adoption of deep learning in inverse problems** by lowering the entrance bar to new researchers and practitioners;
3. Enhance **research reproducibility** via a common framework for imaging operators, reconstruction
methods, datasets, and metrics for inverse problems.

To the best of our knowledge, `deepinv` is the only library with a strong focus and wide set of modern learning-based methods, while covering many imaging domains.
SCICO [@balke2022scico] and Pyxu [@simeoni2022pyxu] focus only on optimization-based methods, and do not include neural network training, diffusion methods, adversarial methods nor unrolling networks.
CUQIpy [@riis2024cuqipy] focuses on Bayesian uncertainty quantification methods for inverse problems.
There are also multiple libraries focusing on specific inverse problems, and we encourage `deepinv` to be used alongside any such library: ASTRA [@van2016astra], pytomography [@polson2025pytomography], TIGRE [@biguri2025tigre], ODL [@adler2018odl] and CIL [@jorgensen2021core] for tomography, sigpy [@ong2019sigpy] for magnetic resonance imaging (without deep learning), and PyLops [@ravasi2019pylops] for certain linear operators. 
Advanced libraries for inverse problems also exist in MATLAB, including GlobalBioIm [@soubies2019pocket] or IR Tools [@gazzola2019ir], but they are restricted to handcrafted reconstruction methods without automatic differentiation.


![Schematic of the library.\label{fig:schematic}](../docs/source/figures/deepinv_schematic.png)

# Inverse Problems

Imaging inverse problems can be expressed as 
\begin{equation} \label{eq:forward}
y = N_{\sigma}(A_{\xi}(x))
\end{equation}
where $x\in\mathcal{X}$ is an image, $y\in\mathcal{Y}$ are the measurements, $A_{\xi}\colon\mathcal{X}\mapsto\mathcal{Y}$ is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
$N_{\sigma}\colon\mathcal{Y}\mapsto \mathcal{Y}$ is a mapping that characterizes the noise affecting the 
measurements (Gaussian, Poisson etc.) parameterized by $\sigma$ (e.g. the noise level and/or gain). The forward operation is written in `deepinv` as `x = physics(y, **params)` where `params` is a dictionary with
optional parameters $\xi$ (blur kernels, noise levels, MRI sampling masks, tomographic projection angles etc.). This framework unifies the wide variety of forward operators across various domains. The ever-expanding library of physics is enumerated in the [documentation](https://deepinv.github.io/deepinv/user_guide/physics/physics.html).

Most forward operators in `deepinv` are matrix-free, scaling gracefully to large image sizes. The framework allows users to perform operator algebra, and provides tools for linear operators such as adjoint, pseudoinverses and proximal operators (leveraging the singular value decomposition when possible), matrix-free linear solvers [@hestenes1952methods] [@paige1975solution] [@van1992bi], and operator norm and condition number estimators [@paige1982lsqr].

The physics `params` $\xi$ also allows for essential advanced problems, including calibration of the system (measuring $\xi$ from $y$),
blind inverse problems (recovering $\xi$ and $x$ from $y$) [@debarnot2024deep] [@chung2023parallel], co-design (optimizing $\xi$ and possibly 
the reconstruction algorithm jointly) [@lazarus2019sparkling] [@nehme2020deepstorm3d], and robust neural network training [@gossard2024training]
[@terris2023meta] [@terris2025ram]. To the best of our knowledge, `deepinv` is unique in offering this feature.


# Reconstruction Methods

`deepinv` unifies the wide variety of commonly-used imaging solvers in the literature:
\begin{equation} \label{eq:solver}
\hat{x} = \operatorname{R}_{\theta}(y, A_{\xi}, \sigma)
\end{equation}
where $\operatorname{R}_{\theta}$ is a reconstruction algorithm with optional trainable parameters $\theta$. The forward pass is written as `x_hat = model(y, physics)`. The ever-expanding library of reconstruction algorithms is enumerated in the [documentation](https://deepinv.github.io/deepinv/user_guide/reconstruction/introduction.html), and can be roughly divided into:

- **Optimization-based**: These solve an optimization problem [@chambolle2016introduction]
\begin{equation} \label{eq:var}
\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) \in \operatorname{argmin}_{x} f_{\sigma}(y,A_{\xi}(x)) + g(x)
\end{equation}.

The `optim` module implements classical data fidelity terms $f_{\sigma}\colon\mathcal{Y} \times \mathcal{Y} \mapsto \mathbb{R}$ and a wide range of regularization priors $g\colon\mathcal{X}\mapsto\mathbb{R}$, including traditional explicit priors [@candes2008introduction] [@rudin1992nonlinear] [@stephane1999wavelet] [@zoran2011learning] [@kowalski2009sparse],
learned regularizers [@zoran2011epll] [@altekruger2023patchnr], and denoising **Plug-and-Play** priors, which replace the proximal operator [@venkatakrishnan2013plug] or the gradient [@romano2017little] of the regularizer $g$ with a pretrained denoiser $\operatorname{D}_{\sigma}$ [@zhang2017beyond] [@zhang2021plug] or recent diffusion-based denoisers [@ho2020denoising] [@song2020score].

To solve these problems, `optim` includes classical algorithms [@dossal2024optimizationorderalgorithms], **unfolded networks** [@gregor2010learning], that unroll a fixed number of iterations of an optimization algorithm and train the parameters end-to-end via backpropagation, and **deep equilibrium methods** [@bai2019deep] that implicitly differentiate the fixed point of the algorithm.

- **Sampling-based**: These are defined via ordinary or stochastic differential equations:
\begin{equation}
x_{t+1} \sim p(x_{t+1}|x_t, y, \operatorname{R}_{\theta}, A_{\xi}, \sigma) \text{ for } t=0,\dots,T-1
\end{equation} 
such that $x_{T}$ is approximately sampled from the posterior $p(x|y)$, and $\operatorname{R}_{\theta}$ is a (potentially learned) denoiser.
Sampling multiple plausible reconstructions enables uncertainty estimates by computing statistics across the samples.

The `sampling` module implements a general modular framework for **diffusion models** [@chung2022diffusion] [@kawar2022denoising] [@zhu2023denoising] and multiple methods of posterior sampling, as well as a framework for popular **Langevin-type algorithms** [@laumont2022bayesian] [@pereyra2020skrock] that sample using Markov Chain Monte Carlo (MCMC) methods with stationary distribution close to the posterior distribution $p(x|y) \propto \exp(f_{\sigma}(y, A_{\xi}(x)) + g(x))$.

- **Non-iterative**: The `models` module implements:
  
  - **Artifact removal** models $\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) = \operatorname{D}_{\sigma}(A_{\xi}^{\top}y),$, which simply backproject the measurements to the image domain and apply a denoiser $\operatorname{D}_{\sigma}$ [@jin2017deep]
  
  - **Generative networks**: both unconditional [@bora2017compressed] [@bora2018ambientgan] [@ulyanov2018deep]
that leverage a pretrained or untrained generator $\operatorname{G}_{\theta}(z)\colon\mathcal{Z}\mapsto \mathcal{X}$ where $z\in\mathcal{Z}$ is a latent code to solve an inverse problem via
$\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) = \operatorname{G}_{\theta}(\hat{z}) \text{ with } \hat{z} \in \operatorname{argmin}_{z} f_{\sigma}(y,A_{\xi}(\operatorname{G}_{\theta}(z)))$,
and conditional [@isola2017image] [@bendel2023gan] that learn a generator $\operatorname{R}_{\theta}(y, z, A_{\xi}, \sigma)$.

  - End-to-end **foundation models** [@terris2025ram] that are trained across a wide variety of forward operators $A_{\xi}$ and noise distributions $N_{\sigma}$. These models often obtain good performance in new tasks without retraining, and can also be finetuned to specific inverse problems or datasets using measurement data alone.

# Training

Reconstruction networks $\operatorname{R}_{\theta}$ can be trained using the `Trainer` high-level interface, handling data ingestion, the training loop, logging, and checkpointing. 

## Losses

The `loss` module framework unifies training loss functions that are widely used in various inverse problems across domains, defined as

\begin{equation} \label{eq:loss}
l = \mathcal{L}\left(\hat{x}, x, y, A_{\xi}, \operatorname{R}_{\theta}\right)
\end{equation}

with $\hat{x}=R_{\theta}(y,A_{\xi},\sigma)$ being the network prediction, 
and written in `deepinv` as `l = loss(x_hat, x, y, physics, model)`. The ever-expanding library of losses is enumerated in the [documentation](https://deepinv.github.io/deepinv/user_guide/training/loss.html).

The **supervised** loss computes a metric between pairs of ground-truth and measurements $\{(x_i,y_i)\}_{i=1}^{N}$. **Self-supervised** losses do not need $x$ and rely on measurement data only $\{y_i\}_{i=1}^{N}$ [@wang2025benchmarking] [@yaman2020self] [@tachella2025unsure] [@pang2021recorrupted] [@monroy2025gr2r] [@chen2021equivariant] [@tachella2022unsupervised].

The `transform` module implements image transforms $\tilde{x}=T_g x$, which can be used for data augmentation, equivariant imaging [@wang2024perspective], and equivariant networks.

**Network regularization** losses enforce some regularity condition on $\operatorname{R}_{\theta}$ [@pesquet2021learning]. **Adversarial** losses train conditional or unconditional generative networks [@bora2017compressed] [@bora2018ambientgan].

## Datasets
The `datasets` module provides a common framework for defining datasets that return ground-truth and measurements pairs $\{(x_i,y_i)\}_{i=1}^{N}$, and optional parameters $\xi_i$, and simulating paired datasets, given a dataset of ground-truth $\{x_i\}_{i=1}^{N}$ and physics, to encourage reproducibility. The ever-expanding library of popular application-specific datasets is enumerated in the [documentation](https://deepinv.github.io/deepinv/user_guide/training/datasets.html).

# Evaluation
The `metric` module provides metrics for evaluating reconstruction methods using `Trainer.test` or for training.
These are written as `m = metric(x_hat, x)` (full-reference), or `m = metric(x_hat)` (no-reference) [@yeganeh2012objective]. `deepinv` includes both distortion [@wang2004image] [@zhang2018unreasonable] and perceptual [@blau2018perception] [@mittal2012making] metrics.

# Philosophy

### Coding Practices

`deepinv` is coded in modern Python following a test-driven development philosophy.
The code is unit-, integration- and performance-tested using `pytest` and verified using `codecov`,
and is compliant with PEP8 using `black`.  To encourage reproducibility, the library passes random number generators
for all random generation functionality. Architecturally, `deepinv` is implemented using an object-oriented framework
where base classes provide abstract functionality and interfaces (such as `Physics` or `Metric`),
sub-classes provide specific implementations or special cases (such as `LinearPhysics`) along with methods inherited
from base classes (such as the operator pseudoinverse), and mixins provide specialized methods. This framework
reduces code duplication and makes it easy for researchers, engineers, and practitioners to implement new or specialized
functionality while inheriting existing methods.

### Documentation

The library provides a [**user guide**](https://deepinv.github.io/deepinv/user_guide.html), which also serves as a tutorial on computational imaging, [quickstart](https://deepinv.github.io/deepinv/quickstart.html) and in-depth [**examples**](https://deepinv.github.io/deepinv/auto_examples/index.html) for all levels of user, and individual [API documentation](https://deepinv.github.io/deepinv/API.html)
for classes. The documentation is generated using Sphinx and Sphinx-Gallery [@najera2023sphinxgallery], tested using `doctest`, and uses consistent mathematical notation throughout.

# Perspectives

DeepInverse is a dynamic and evolving project and this paper is merely a snapshot of ongoing progress (release v0.3.2). The community is continuously contributing more methods, such as more realistic physics operators and more advanced training techniques, which reflect the state-of-the-art in imaging with deep learning, addressing the needs and interests of researchers and practitioners alike.

# Acknowledgements

J. Tachella acknowledges support by the French National Research Agency (Agence Nationale de la Recherche) grant UNLIP (ANR-23-CE23-0013) and the CNRS PNRIA deepinverse project.
M. Terris acknowledges support by the BrAIN grant (ANR-20-CHIA-0016).
F. Sarron, P. Weiss, M.H. Nguyen were supported by the ANR Micro-Blind ANR-21-CE48-0008.
T. Moreau was supported from a national grant attributed to the ExaDoST project of the NumPEx PEPR program, under the reference ANR-22-EXNU-0004.
J. Hertrich is supported by the German Research Foundation (DFG) with project number 530824055.
Z. Hu acknowledges funding from the Swiss National Science Foundation (grant PZ00P2_216211). Thomas Davies is supported by UKRI EPSRC (grants EP/V006134/1 and EP/V006177/1).
S. Neumayer acknowledges funding from the German Research Foundation (DFG) with project number 543939932.
We thank the [BASP Laboratory at Heriot-Watt University](https://basp.site.hw.ac.uk/) for insightful discussions, and their contribution on the radio astronomy application.
The authors acknowledge the Jean-Zay high-performance computing center, using HPC resources from GENCI-IDRIS (Grants 2021-AD011012210, 2024-AD011015191). 

# References
