# deepinv.sampling

This module contains various posterior sampling algorithms, including diffusion-based methods and MCMC methods.
Please refer to the [user guide](https://deepinv.org/user_guide/reconstruction/sampling.md#sampling) for more details.

## Diffusion models with Stochastic Differential Equations for Image Generation and Posterior Sampling

**User Guide:** refer to [Diffusion models](https://deepinv.org/user_guide/reconstruction/sampling.md#diffusion) for more information.

| [`deepinv.sampling.BaseSDE`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.md#deepinv.sampling.BaseSDE)                                         | Base class for Stochastic Differential Equation (SDE):                                                                                                                               |
|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.md#deepinv.sampling.DiffusionSDE)                               | Define the Reverse-time Diffusion Stochastic Differential Equation.                                                                                                                  |
| [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.md#deepinv.sampling.EDMDiffusionSDE)                         | Generative diffusion Stochastic Differential Equation.                                                                                                                               |
| [`deepinv.sampling.SongDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.SongDiffusionSDE.md#deepinv.sampling.SongDiffusionSDE)                       | Generative diffusion Stochastic Differential Equation.                                                                                                                               |
| [`deepinv.sampling.FlowMatching`](https://deepinv.org/api/stubs/deepinv.sampling.FlowMatching.md#deepinv.sampling.FlowMatching)                               | Generative Flow Matching process.                                                                                                                                                    |
| [`deepinv.sampling.VarianceExplodingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VarianceExplodingDiffusion.md#deepinv.sampling.VarianceExplodingDiffusion)   |                                                                                                                                                                                      |
| [`deepinv.sampling.VariancePreservingDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.md#deepinv.sampling.VariancePreservingDiffusion) | Variance-Preserving Stochastic Differential Equation (VP-SDE).                                                                                                                       |
| [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.md#deepinv.sampling.PosteriorDiffusion)                   | Posterior distribution sampling  for inverse problems using diffusion models by Reverse-time Stochastic Differential Equation (SDE).                                                 |
| [`deepinv.sampling.NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.md#deepinv.sampling.NoisyDataFidelity)                     | Data fidelity term for noisy input data $- \log p(y|x + \sigma(t) \omega)$ with $\omega\sim\mathcal{N}(0,\mathrm{I})$.                                                               |
| [`deepinv.sampling.ALDDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.ALDDataFidelity.md#deepinv.sampling.ALDDataFidelity)                         | Score-based annealed Langevin dynamics (Score-ALD) data-fidelity term.                                                                                                               |
| [`deepinv.sampling.ScoreSDEDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.ScoreSDEDataFidelity.md#deepinv.sampling.ScoreSDEDataFidelity)               | Score-SDE data-fidelity term.                                                                                                                                                        |
| [`deepinv.sampling.ILVRDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.ILVRDataFidelity.md#deepinv.sampling.ILVRDataFidelity)                       | Iterative Latent Variable Refinement (ILVR) data-fidelity term.                                                                                                                      |
| [`deepinv.sampling.DPSDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DPSDataFidelity.md#deepinv.sampling.DPSDataFidelity)                         | Diffusion posterior sampling data-fidelity term.                                                                                                                                     |
| [`deepinv.sampling.PiGDMDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.PiGDMDataFidelity.md#deepinv.sampling.PiGDMDataFidelity)                     | Pseudoinverse-guided diffusion model (PiGDM) data-fidelity term.                                                                                                                     |
| [`deepinv.sampling.MomentMatchingDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.MomentMatchingDataFidelity.md#deepinv.sampling.MomentMatchingDataFidelity)   | Moment-matching data-fidelity term for diffusion posterior sampling.                                                                                                                 |
| [`deepinv.sampling.BaseSDESolver`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.md#deepinv.sampling.BaseSDESolver)                             | Base class for solving Stochastic Differential Equations (SDEs) from [`deepinv.sampling.BaseSDE`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.md#deepinv.sampling.BaseSDE) of the form: |
| [`deepinv.sampling.EulerSolver`](https://deepinv.org/api/stubs/deepinv.sampling.EulerSolver.md#deepinv.sampling.EulerSolver)                                 | Euler-Maruyama solver for SDEs.                                                                                                                                                      |
| [`deepinv.sampling.HeunSolver`](https://deepinv.org/api/stubs/deepinv.sampling.HeunSolver.md#deepinv.sampling.HeunSolver)                                   | Heun solver for SDEs.                                                                                                                                                                |
| [`deepinv.sampling.SDEOutput`](https://deepinv.org/api/stubs/deepinv.sampling.SDEOutput.md#deepinv.sampling.SDEOutput)                                     | A container for storing the output of an SDE solver, that behaves like a `dict` but allows access with the attribute syntax.                                                         |

## Custom diffusion posterior samplers

**User Guide:** refer to [Popular posterior samplers](https://deepinv.org/user_guide/reconstruction/sampling.md#diffusion-custom) for more information.

| [`deepinv.sampling.DDRM`](https://deepinv.org/api/stubs/deepinv.sampling.DDRM.md#deepinv.sampling.DDRM)                         | Denoising Diffusion Restoration Models (DDRM).       |
|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| [`deepinv.sampling.DiffPIR`](https://deepinv.org/api/stubs/deepinv.sampling.DiffPIR.md#deepinv.sampling.DiffPIR)                   | Diffusion PnP Image Restoration (DiffPIR).           |
| [`deepinv.sampling.DPS`](https://deepinv.org/api/stubs/deepinv.sampling.DPS.md#deepinv.sampling.DPS)                           | Diffusion Posterior Sampling (DPS).                  |
| [`deepinv.sampling.DiffusionSampler`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSampler.md#deepinv.sampling.DiffusionSampler) | Turns a diffusion method into a Monte Carlo sampler. |

## Base Class

**User Guide:** refer to [Diffusion and MCMC Algorithms](https://deepinv.org/user_guide/reconstruction/sampling.md#sampling) for more information.

| [`deepinv.sampling.sampling_builder`](https://deepinv.org/api/stubs/deepinv.sampling.sampling_builder.md#deepinv.sampling.sampling_builder)   | Helper function for building an instance of the [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.md#deepinv.sampling.BaseSampling) class.   |
|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

| [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.md#deepinv.sampling.BaseSampling)   | Base class for Monte Carlo sampling.   |
|----------------------------------------------------------------------------------------------------------------|----------------------------------------|

## Markov Chain Monte Carlo Langevin

**User Guide:** refer to [Markov Chain Monte Carlo](https://deepinv.org/user_guide/reconstruction/sampling.md#mcmc) for more information.

| [`deepinv.sampling.ULA`](https://deepinv.org/api/stubs/deepinv.sampling.ULA.md#deepinv.sampling.ULA)       | Projected Plug-and-Play Unadjusted Langevin Algorithm.   |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [`deepinv.sampling.SKRock`](https://deepinv.org/api/stubs/deepinv.sampling.SKRock.md#deepinv.sampling.SKRock) | Plug-and-Play SKROCK algorithm.                          |

## Iterators

| [`deepinv.sampling.SamplingIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.md#deepinv.sampling.SamplingIterator)   | Base class for sampling iterators.                                                                                                                                                                                                                                                                       |
|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.sampling.SKRockIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SKRockIterator.md#deepinv.sampling.SKRockIterator)       | Single iteration of the SK-ROCK (Stabilized Runge-Kutta-Chebyshev) Algorithm.                                                                                                                                                                                                                            |
| [`deepinv.sampling.ULAIterator`](https://deepinv.org/api/stubs/deepinv.sampling.ULAIterator.md#deepinv.sampling.ULAIterator)             | Projected Plug-and-Play Unadjusted Langevin Algorithm.                                                                                                                                                                                                                                                   |
| [`deepinv.sampling.DiffusionIterator`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionIterator.md#deepinv.sampling.DiffusionIterator) | Helper class used by [`deepinv.sampling.DiffusionSampler`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSampler.md#deepinv.sampling.DiffusionSampler) to interface diffusion models with the [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.md#deepinv.sampling.BaseSampling) framework. |
