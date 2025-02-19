---
title: 'deepinverse: A Python package for solving imaging inverse problems with deep learning'
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

# Inverse problems

Imaging inverse problems can be expressed as 
\begin{equation} \label{eq:solver}
y = N(A(x))
\end{equation} 
which in deepinv code is simply written as `x_hat = physics(y)`.


.. list-table:: Operators, Definitions, and Generators
   :header-rows: 1

   * - **Family**
     - **Operators**
     - **Generators**

   * - Pixelwise
     -
       | `deepinv.physics.Denoising`
       | `deepinv.physics.Inpainting`
       | `deepinv.physics.Demosaicing`
       | `deepinv.physics.Decolorize`
     -
       | `BernoulliSplittingMaskGenerator <deepinv.physics.generator.BernoulliSplittingMaskGenerator>`
       | `GaussianSplittingMaskGenerator <deepinv.physics.generator.GaussianSplittingMaskGenerator>`
       | `Phase2PhaseSplittingMaskGenerator <deepinv.physics.generator.Phase2PhaseSplittingMaskGenerator>`
       | `Artifact2ArtifactSplittingMaskGenerator <deepinv.physics.generator.Artifact2ArtifactSplittingMaskGenerator>`

   * - Blur & Super-Resolution
     -
       | `deepinv.physics.Blur`
       | `deepinv.physics.BlurFFT`
       | `deepinv.physics.SpaceVaryingBlur`
       | `deepinv.physics.Downsampling`
     -
       | `MotionBlurGenerator <deepinv.physics.generator.MotionBlurGenerator>`
       | `DiffractionBlurGenerator <deepinv.physics.generator.DiffractionBlurGenerator>`
       | `ProductConvolutionBlurGenerator <deepinv.physics.generator.ProductConvolutionBlurGenerator>`
       | `ConfocalBlurGenerator3D <deepinv.physics.generator.ConfocalBlurGenerator3D>`
       | `gaussian_blur <deepinv.physics.blur.gaussian_blur>`, `sinc_filter <deepinv.physics.blur.sinc_filter>`
       | `bilinear_filter <deepinv.physics.blur.bilinear_filter>`, `bicubic_filter <deepinv.physics.blur.bicubic_filter>`

   * - Magnetic Resonance Imaging (MRI)
     -
       | `deepinv.physics.MRIMixin`
       | `deepinv.physics.MRI`
       | `deepinv.physics.MultiCoilMRI`
       | `deepinv.physics.DynamicMRI`
       | `deepinv.physics.SequentialMRI`
       | The above all also natively support 3D MRI.
     -
       | `GaussianMaskGenerator <deepinv.physics.generator.GaussianMaskGenerator>`
       | `RandomMaskGenerator <deepinv.physics.generator.RandomMaskGenerator>`
       | `EquispacedMaskGenerator <deepinv.physics.generator.EquispacedMaskGenerator>`
       | The above all also support k+t dynamic sampling.

   * - Tomography
     -
       | `deepinv.physics.Tomography`
     -

   * - Remote Sensing & Multispectral
     -
       | `deepinv.physics.Pansharpen`
       | `deepinv.physics.HyperSpectralUnmixing`
       | `deepinv.physics.CompressiveSpectralImaging`
     -

   * - Compressive
     -
       | `deepinv.physics.CompressedSensing`
       | `deepinv.physics.StructuredRandom`
       | `deepinv.physics.SinglePixelCamera`
     -

   * - Radio Interferometric Imaging
     -
       | `deepinv.physics.RadioInterferometry`
     -

   * - Single-Photon Lidar
     -
       | `deepinv.physics.SinglePhotonLidar`
     -

   * - Dehazing
     -
       | `deepinv.physics.Haze`
     -

   * - Phase Retrieval
     -
       | `deepinv.physics.PhaseRetrieval`
       | `RandomPhaseRetrieval <deepinv.physics.RandomPhaseRetrieval>`
       | `StructuredRandomPhaseRetrieval <deepinv.physics.StructuredRandomPhaseRetrieval>`
       | `Ptychography <deepinv.physics.Ptychography>`
       | `PtychographyLinearOperator <deepinv.physics.PtychographyLinearOperator>`
     - | `build_probe <deepinv.physics.phase_retrieval.build_probe>`
       | `generate_shifts <deepinv.physics.phase_retrieval.generate_shifts>`

       
# Reconstruction methods

The library provides multiple solvers which depend on the acquisition physics:
\begin{equation} \label{eq:solver}
\hat{x} = R_{\theta}(y, A)
\end{equation}
which in deepinv code is simply written as `x_hat = solver(y, A)`.


.. list-table:: Reconstruction methods
   :header-rows: 1

   * - **Family of methods**
     - **Description**
     - **Requires Training**
     - **Iterative**
     - **Sampling**
   * - `Artifact Removal <artifact>`
     - Applies a neural network to a non-learned pseudo-inverse
     - Yes
     - No
     - No
   * - `Plug-and-Play (PnP) <iterative>`
     - Leverages `pretrained denoisers <denoisers>` as priors within an optimisation algorithm.
     - No
     - Yes
     - No
   * - `Unfolded Networks <unfolded>`
     - Constructs a trainable architecture by unrolling a PnP algorithm.
     - Yes
     - Only `DEQ <deepinv.unfolded.DEQ_builder>`
     - No
   * - `Diffusion <diffusion>`
     - Leverages `pretrained denoisers <denoisers>` within a ODE/SDE.
     - No
     - Yes
     - Yes
   * - `Non-learned priors <iterative>`
     - Solves an optimization problem with hand-crafted priors.
     - No
     - Yes
     - No
   * - `Markov Chain Monte Carlo <mcmc>`
     - Leverages `pretrained denoisers <denoisers>` as priors within an optimisation algorithm.
     - No
     - Yes
     - Yes
   * - `Generative Adversarial Networks and Deep Image Prior  <adversarial>`
     - Uses a generator network to model the set of possible images.
     - No
     - Yes
     - Depends
   * - `Specific network architectures <specific>`
     - Off-the-shelf architectures for specific inverse problems.
     - Yes
     - No
     - No


![Schematic of the library.\label{fig:schematic}](figures/deepinv_schematic_.png)
referenced from text using \autoref{fig:schematic}.


# Training

# Acknowledgements

Julián Tachella acknowledges support by the ANR grant UNLIP (ANR-23-CE23-0013) and the CNRS PNRIA deepinverse project.
The authors would acknowledge the Jean-Zay high-performance computing center for providing computational resources.

# References