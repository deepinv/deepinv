# deepinv.sampling

This module contains various posterior sampling algorithms, including diffusion-based methods and MCMC methods.
Please refer to the [user guide](https://deepinv.org/user_guide/reconstruction/sampling.html.md#sampling) for more details.

## Diffusion models with Stochastic Differential Equations for Image Generation and Posterior Sampling

**User Guide:** refer to [Diffusion models](https://deepinv.org/user_guide/reconstruction/sampling.html.md#diffusion) for more information.

| [`deepinv.sampling.BaseSDE`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.html.md#deepinv.sampling.BaseSDE)                                         | Base class for Stochastic Differential Equation (SDE):                                                                                                                               |
|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE)                               | Define the Reverse-time Diffusion Stochastic Differential Equation.                                                                                                                  |
| [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE)                         | Generative diffusion Stochastic Differential Equation.                                                                                                                               |
| [`deepinv.sampling.SongDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.SongDiffusionSDE.html.md#deepinv.sampling.SongDiffusionSDE)                       | Generative diffusion Stochastic Differential Equation.                                                                                                                               |
| [`deepinv.sampling.FlowMatching`](https://deepinv.org/api/stubs/deepinv.sampling.FlowMatching.html.md#deepinv.sampling.FlowMatching)                               | Generative Flow Matching process.                                                                                                                                                    |
| [`deepinv.sampling.VarianceExplodingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VarianceExplodingDiffusion.html.md#deepinv.sampling.VarianceExplodingDiffusion)   |                                                                                                                                                                                      |
| [`deepinv.sampling.VariancePreservingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.html.md#deepinv.sampling.VariancePreservingDiffusion) | Variance-Preserving Stochastic Differential Equation (VP-SDE).                                                                                                                       |
| [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion)                   | Posterior distribution sampling  for inverse problems using diffusion models by Reverse-time Stochastic Differential Equation (SDE).                                                 |
| [`deepinv.sampling.NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.html.md#deepinv.sampling.NoisyDataFidelity)                     | Preconditioned data fidelity term for noisy data $- \log p(y|x + \sigma(t) \omega)$ with $\omega\sim\mathcal{N}(0,\mathrm{I})$.                                                      |
| [`deepinv.sampling.DPSDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DPSDataFidelity.html.md#deepinv.sampling.DPSDataFidelity)                         | Diffusion posterior sampling data-fidelity term.                                                                                                                                     |
| [`deepinv.sampling.BaseSDESolver`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)                             | Base class for solving Stochastic Differential Equations (SDEs) from [`deepinv.sampling.BaseSDE`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.html.md#deepinv.sampling.BaseSDE) of the form: |
| [`deepinv.sampling.EulerSolver`](https://deepinv.org/api/stubs/deepinv.sampling.EulerSolver.html.md#deepinv.sampling.EulerSolver)                                 | Euler-Maruyama solver for SDEs.                                                                                                                                                      |
| [`deepinv.sampling.HeunSolver`](https://deepinv.org/api/stubs/deepinv.sampling.HeunSolver.html.md#deepinv.sampling.HeunSolver)                                   | Heun solver for SDEs.                                                                                                                                                                |
| [`deepinv.sampling.SDEOutput`](https://deepinv.org/api/stubs/deepinv.sampling.SDEOutput.html.md#deepinv.sampling.SDEOutput)                                     | A container for storing the output of an SDE solver, that behaves like a `dict` but allows access with the attribute syntax.                                                         |

## Custom diffusion posterior samplers

**User Guide:** refer to [Popular posterior samplers](https://deepinv.org/user_guide/reconstruction/sampling.html.md#diffusion-custom) for more information.

| [`deepinv.sampling.DDRM`](https://deepinv.org/api/stubs/deepinv.sampling.DDRM.html.md#deepinv.sampling.DDRM)                         | Denoising Diffusion Restoration Models (DDRM).       |
|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| [`deepinv.sampling.DiffPIR`](https://deepinv.org/api/stubs/deepinv.sampling.DiffPIR.html.md#deepinv.sampling.DiffPIR)                   | Diffusion PnP Image Restoration (DiffPIR).           |
| [`deepinv.sampling.DPS`](https://deepinv.org/api/stubs/deepinv.sampling.DPS.html.md#deepinv.sampling.DPS)                           | Diffusion Posterior Sampling (DPS).                  |
| [`deepinv.sampling.DiffusionSampler`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSampler.html.md#deepinv.sampling.DiffusionSampler) | Turns a diffusion method into a Monte Carlo sampler. |

## Base Class

**User Guide:** refer to [Diffusion and MCMC Algorithms](https://deepinv.org/user_guide/reconstruction/sampling.html.md#sampling) for more information.

| [`deepinv.sampling.sampling_builder`](https://deepinv.org/api/stubs/deepinv.sampling.sampling_builder.html.md#deepinv.sampling.sampling_builder)   | Helper function for building an instance of the [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling) class.   |
|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

| [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling)   | Base class for Monte Carlo sampling.   |
|----------------------------------------------------------------------------------------------------------------|----------------------------------------|

## Markov Chain Monte Carlo Langevin

**User Guide:** refer to [Markov Chain Monte Carlo](https://deepinv.org/user_guide/reconstruction/sampling.html.md#mcmc) for more information.

| [`deepinv.sampling.ULA`](https://deepinv.org/api/stubs/deepinv.sampling.ULA.html.md#deepinv.sampling.ULA)       | Projected Plug-and-Play Unadjusted Langevin Algorithm.   |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [`deepinv.sampling.SKRock`](https://deepinv.org/api/stubs/deepinv.sampling.SKRock.html.md#deepinv.sampling.SKRock) | Plug-and-Play SKROCK algorithm.                          |

## Iterators

| [`deepinv.sampling.SamplingIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator)   | Base class for sampling iterators.                                                                                                                                                                                                                                                                       |
|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.sampling.SKRockIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SKRockIterator.html.md#deepinv.sampling.SKRockIterator)       | Single iteration of the SK-ROCK (Stabilized Runge-Kutta-Chebyshev) Algorithm.                                                                                                                                                                                                                            |
| [`deepinv.sampling.ULAIterator`](https://deepinv.org/api/stubs/deepinv.sampling.ULAIterator.html.md#deepinv.sampling.ULAIterator)             | Projected Plug-and-Play Unadjusted Langevin Algorithm.                                                                                                                                                                                                                                                   |
| [`deepinv.sampling.DiffusionIterator`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionIterator.html.md#deepinv.sampling.DiffusionIterator) | Helper class used by [`deepinv.sampling.DiffusionSampler`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSampler.html.md#deepinv.sampling.DiffusionSampler) to interface diffusion models with the [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling) framework. |
