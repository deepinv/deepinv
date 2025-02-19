---
title: 'deepinverse: A Python package for solving imaging inverse problems with deep learning'
tags:
  - Python
  - Pytorch
  - imaging inverse problems
  - computational imaging
authors:
  - name: Julian Tachella
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Matthieu Terris
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Samuel Hurault
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 3
  - name: Andrew Wang
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 4
  - name: Dongdong Chen
  - affiliation: 5
  - name: Contributors (TODO)
  - affiliation: 5

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

DeepInverse is an open-source PyTorch-based library for solving imaging inverse problems with deep learning.
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
where 

The library provides multiple solvers which depend on the acquisition physics:
\begin{equation} \label{eq:solver}
\hat{x} = \text{solver}(y, A)
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

# Figures

Figures can be included like this:
![Schematic of the library.\label{fig:schematic}](figures/deepinv_schematic_.png)
and referenced from text using \autoref{fig:example}.

# Acknowledgements

We acknowledge contributions from ...

# References