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
  - name: CNRS, ENS de Lyon, France
    index: 1
  - name: Université Paris-Saclay, Inria, CEA, Palaiseau, France
    index: 2
  - name: CNRS, ENS Paris, PSL, France
    index: 3
  - name: University of Edinburgh, UK
    index: 4
  - name: Heriot-Watt University, Edinburgh, UK
    index: 5
  - name: Universidad Industrial de Santander, Bucaramanga, Colombia
    index: 6
  - name: EPFL, Lausanne, Switzerland
    index: 7
  - name: IRIT, CBI, CNRS, Université de Toulouse, France
    index: 8
  - name: IMT, CNRS, Université de Toulouse, France
    index: 9
  - name: Inria, ENS de Lyon, France
    index: 10
  - name: INSA de Lyon, France
    index: 11
  - name: University of Texas at Austin, USA
    index: 12
  - name: IRFU, CEA, Université Paris-Saclay, Gif-sur-Yvette, France
    index: 13
  - name: CNRS UAR 851, Université Paris-Saclay Orsay, France
    index: 14
  - name: Université Paris Dauphine - PSL, Paris, France
    index: 15
  - name: Chemnitz University of Technology, Chemnitz, Germany
    index: 16
  - name: Department of Imaging and Pathology, KU Leuven, Belgium
    index: 17

date: 15 May 2025
bibliography: paper.bib

---

# Summary

[DeepInverse](https://deepinv.github.io/) is an open-source PyTorch-based library for imaging inverse problems. DeepInverse implements all the steps to reconstruct images using deep learning for a wide set of domains (medical imaging, astronomical imaging, remote sensing, computational photography, compressed sensing and more).

# Statement of Need

Deep neural networks have become ubiquitous in various imaging inverse problems. Despite the ever-increasing research effort, most learning-based algorithms are built from scratch, are hard to generalize beyond their specific training setting, and the reported results are often hard to reproduce. `deepinv` overcomes these limitations by providing a modular unified framework, leveraging the popular PyTorch deep learning library [@paszke2019pytorch].
For our audience of researchers (experts in optimization, deep learning etc.), practitioners (biologists, physicists etc.) and imaging software engineers, `deepinv` is:

1. **Accelerating research** by enabling efficient testing, deployment and transfer
of new ideas across imaging domains;
2. Enlarging the **adoption of deep learning in inverse problems** by lowering the entrance bar to new users;
3. Enhancing **research reproducibility** via a common modular framework of problems and algorithms.

To the best of our knowledge, `deepinv` is the only library with a strong focus and wide set of modern learning-based methods across domains.
SCICO [@balke2022scico] and Pyxu [@simeoni2022pyxu] focus on optimization-based methods.
CUQIpy [@riis2024cuqipy] focuses on Bayesian uncertainty quantification.
ASTRA [@van2016astra], pytomography [@polson2025pytomography], TIGRE [@biguri2025tigre], ODL [@adler2018odl] and CIL [@jorgensen2021core] focus on tomography, sigpy [@ong2019sigpy] on magnetic resonance imaging, and PyLops [@ravasi2019pylops] on certain operators. 
MATLAB libraries ([@soubies2019pocket], IR Tools [@gazzola2019ir]) are restricted to handcrafted methods without automatic differentiation.

![Schematic of the library.\label{fig:schematic}](../docs/source/figures/deepinv_schematic.png)

# Inverse Problems

Imaging inverse problems can be expressed as 
\begin{equation} \label{eq:forward}
y = N_{\sigma}(A_{\xi}(x))
\end{equation}
where $x\in\mathcal{X}$ is an image, $y\in\mathcal{Y}$ are the measurements, $A_{\xi}\colon\mathcal{X}\mapsto\mathcal{Y}$ is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
$N_{\sigma}\colon\mathcal{Y}\mapsto \mathcal{Y}$ is a noise model parameterized by $\sigma$. The [`physics` module](https://deepinv.github.io/deepinv/user_guide/physics/intro.html) provides a scalable and modular framework, writing the forward operation as `x = physics(y, **params)`, unifying the wide variety of forward operators across various domains. 

The optional physics `params` $\xi$ allows for advanced problems, including calibration,
blind inverse problems [@debarnot2024deep] [@chung2023parallel], codesign [@lazarus2019sparkling] [@nehme2020deepstorm3d], and robust training [@gossard2024training] [@terris2023meta].

The ever-expanding library of physics, noise models, parameters $\xi$ and tools for manipulating them are enumerated in the [documentation](https://deepinv.github.io/deepinv/user_guide/physics/physics.html).


# Reconstruction Methods

`deepinv` unifies the wide variety of commonly-used imaging solvers in the literature:
\begin{equation} \label{eq:solver}
\hat{x} = \operatorname{R}_{\theta}(y, A_{\xi}, \sigma)
\end{equation}
where $\operatorname{R}_{\theta}$ is a reconstruction algorithm with optional trainable parameters $\theta$ and $\hat{x}$ is the reconstructed image, written as `x_hat = model(y, physics)`. The ever-expanding library of algorithms is enumerated in the [documentation](https://deepinv.github.io/deepinv/user_guide/reconstruction/introduction.html), categorized as:

- **Optimization-based** methods [@chambolle2016introduction] solve
\begin{equation} \label{eq:var}
\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) \in \operatorname{argmin}_{x} f_{\sigma}(y,A_{\xi}(x)) + g(x)
\end{equation}.

  The [`optim` module](https://deepinv.github.io/deepinv/user_guide/reconstruction/optimization.html) implements classical data fidelity terms $f_{\sigma}\colon\mathcal{Y} \times \mathcal{Y} \mapsto \mathbb{R}$ and a variety of regularization priors $g\colon\mathcal{X}\mapsto\mathbb{R}$, including:
  - Traditional explicit priors [@candes2008introduction];
  - Learned regularizers [@zoran2011epll] [@altekruger2023patchnr];
  - **Plug-and-Play** priors [@venkatakrishnan2013plug] [@romano2017little] using a pretrained denoiser $\operatorname{D}_{\sigma}$ [@zhang2017beyond] [@zhang2021plug] [@ho2020denoising] [@song2020score].

  To solve these problems, `optim` includes:
  - Classical algorithms [@dossal2024optimizationorderalgorithms];
  - **Unfolded networks** [@gregor2010learning], that unroll a fixed number of iterations of an optimization algorithm and train the parameters end-to-end;
  - **Deep equilibrium methods** [@bai2019deep] that implicitly differentiate the fixed point of the algorithm.

- **Sampling-based** methods defined by differential equations:
\begin{equation}
x_{t+1} \sim p(x_{t+1}|x_t, y, \operatorname{R}_{\theta}, A_{\xi}, \sigma) \text{ for } t=0,\dots,T-1
\end{equation} 
such that $x_{T}$ is approximately sampled from the posterior $p(x|y)$, and $\operatorname{R}_{\theta}$ is a denoiser. Sampling multiple times enables uncertainty quantification.

  The [`sampling` module](https://deepinv.github.io/deepinv/user_guide/reconstruction/sampling.html) implements general modular frameworks for:
  - **Diffusion model** posterior sampling [@chung2022diffusion] [@kawar2022denoising] [@zhu2023denoising];
  - **Langevin-type algorithms** [@laumont2022bayesian] [@pereyra2020skrock] that sample using Markov Chain Monte Carlo.

- **Non-iterative**: The [`models` module](https://deepinv.github.io/deepinv/user_guide/reconstruction/introduction.html) implements:
  
  - **Artifact removal** models $\operatorname{R}_{\theta}(y, A_{\xi}, \sigma) = \operatorname{D}_{\sigma}(A_{\xi}^{\top}y),$, which simply backproject $y$ to the image domain and apply a denoiser $\operatorname{D}_{\sigma}$ [@jin2017deep];
  
  - **Generative networks** [@bora2018ambientgan] [@bendel2023gan] [@ulyanov2018deep]
that add a latent $z$ to a generator network $\operatorname{G}_{\theta}(y,z)\colon\mathcal{Y}\times\mathcal{Z}\mapsto \mathcal{X}$;

  - **Foundation models** [@terris2025ram], trained end-to-end across a wide variety of $A_{\xi},N_{\sigma}$, and can be finetuned to new problems.

# Training

Reconstruction networks $\operatorname{R}_{\theta}$ can be trained using the [`Trainer` high-level interface](https://deepinv.github.io/deepinv/user_guide/training/trainer.html).

## Losses

The [`loss` module](https://deepinv.github.io/deepinv/user_guide/training/loss.html) framework unifies training loss functions that are widely used across various domains:

\begin{equation} \label{eq:loss}
l = \mathcal{L}\left(\hat{x}, x, y, A_{\xi}, \operatorname{R}_{\theta}\right)
\end{equation}

The ever-expanding library of losses `loss(x_hat, x, y, physics, model)` is enumerated in the [documentation](https://deepinv.github.io/deepinv/user_guide/training/loss.html):
- **Supervised** loss between $x$ and $y$;
- **Self-supervised** losses which only use $y$ [@wang2025benchmarking] [@yaman2020self] [@tachella2025unsure] [@monroy2025gr2r] [@chen2021equivariant] [@tachella2022unsupervised];
- **Network regularization** losses [@pesquet2021learning];
- **Adversarial** losses [@bora2017compressed] [@bora2018ambientgan].

The [`transform` module](https://deepinv.github.io/deepinv/user_guide/training/transforms.html) implements transforms $\tilde{x}=T_g x$ for data augmentation and equivariance [@wang2024perspective].

## Datasets
The [`datasets` module](https://deepinv.github.io/deepinv/user_guide/training/datasets.html) implements a variety of domain-specific datasets that return ground-truth and measurements pairs $\{(x_i,y_i)\}_{i=1}^{N}$ and optional parameters $\xi_i$, and allows simulating paired datasets given $\{x_i\}_{i=1}^{N}$ and physics.

# Evaluation
The [`metric` module](https://deepinv.github.io/deepinv/user_guide/training/metric.html#metric) provides metrics for evaluating reconstruction methods using `Trainer.test` or for training.
These are written as `m = metric(x_hat, x)` (full-reference), or `m = metric(x_hat)` (no-reference) [@yeganeh2012objective], including distortion [@wang2004image] [@zhang2018unreasonable] and perceptual [@blau2018perception] [@mittal2012making] metrics.

# Philosophy

### Coding Practices

`deepinv` uses modern test-driven Python.
The code is unit-, integration- and performance-tested using `pytest` and verified using `codecov`,
and is compliant with PEP8 using `black`. We adopt an object-oriented framework
where base classes provide abstract functionality and interfaces,
subclasses provide specific method implementations or special cases, reducing code duplication, facilitating users to implement new or specialized functionality while inheriting existing methods.

### Documentation

The library provides a [**user guide**](https://deepinv.github.io/deepinv/user_guide.html), which also serves as a tutorial on computational imaging, [quickstart](https://deepinv.github.io/deepinv/quickstart.html) and in-depth [**examples**](https://deepinv.github.io/deepinv/auto_examples/index.html) for all levels of user, and individual [API documentation](https://deepinv.github.io/deepinv/API.html)
for classes. The documentation is generated using Sphinx and Sphinx-Gallery [@najera2023sphinxgallery], tested using `doctest`, and uses consistent mathematical notation throughout.

# Perspectives

DeepInverse is a dynamic and evolving project and this paper is merely a snapshot of ongoing progress. The community is continuously contributing more methods reflecting state-of-the-art in imaging with deep learning, addressing the needs and interests of researchers and practitioners.

# Acknowledgements

J. Tachella acknowledges support by the French ANR grant UNLIP (ANR-23-CE23-0013) and the CNRS PNRIA DeepInverse project.
M. Terris acknowledges support by the BrAIN grant (ANR-20-CHIA-0016).
F. Sarron, P. Weiss, M.H. Nguyen were supported by ANR Micro-Blind ANR-21-CE48-0008.
T. Moreau was supported by the ExaDoST project under NumPEx PEPR (ANR-22-EXNU-0004).
J. Hertrich is supported by DFG (project 530824055).
Z. Hu acknowledges funding from the Swiss National Science Foundation (grant PZ00P2_216211).
T. Davies is supported by UKRI EPSRC (grants EP/V006134/1, EP/V006177/1).
S. Neumayer acknowledges funding from DFG (project 543939932).
We thank the [BASP Laboratory at Heriot-Watt University](https://basp.site.hw.ac.uk/) for insightful discussions and contributions to the radioastronomy application.
The authors acknowledge the Jean-Zay HPC (GENCI-IDRIS grants 2021-AD011012210, 2024-AD011015191). 

# References
