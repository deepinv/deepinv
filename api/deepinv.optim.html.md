# deepinv.optim

This module provides optimization utils for constructing reconstruction models based on optimization algorithms.
Please refer to the [user guide](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim) for more details.

## Base Class

**User Guide:** refer to [Optimization](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim) for more information.

| [`deepinv.optim.optim_builder`](https://deepinv.org/api/stubs/deepinv.optim.optim_builder.html.md#deepinv.optim.optim_builder)   | Helper function for building an instance of the [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) class.   |
|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|

| [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)                                   | Class for optimization algorithms, consists in iterating a fixed-point operator.             |
|------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [`deepinv.optim.BacktrackingConfig`](https://deepinv.org/api/stubs/deepinv.optim.BacktrackingConfig.html.md#deepinv.optim.BacktrackingConfig)                 | Configuration parameters for backtracking line search on the stepsize.                       |
| [`deepinv.optim.AndersonAccelerationConfig`](https://deepinv.org/api/stubs/deepinv.optim.AndersonAccelerationConfig.html.md#deepinv.optim.AndersonAccelerationConfig) | Configuration parameters for Anderson acceleration of a fixed-point algorithm.               |
| [`deepinv.optim.DEQConfig`](https://deepinv.org/api/stubs/deepinv.optim.DEQConfig.html.md#deepinv.optim.DEQConfig)                                   | Configuration parameters for Deep Equilibrium models.                                        |
| [`deepinv.optim.GD`](https://deepinv.org/api/stubs/deepinv.optim.GD.html.md#deepinv.optim.GD)                                                 | Gradient Descent (GD) module for solving the problem                                         |
| [`deepinv.optim.PGD`](https://deepinv.org/api/stubs/deepinv.optim.PGD.html.md#deepinv.optim.PGD)                                               | Proximal Gradient Descent (PGD) module for solving the problem                               |
| [`deepinv.optim.FISTA`](https://deepinv.org/api/stubs/deepinv.optim.FISTA.html.md#deepinv.optim.FISTA)                                           | FISTA module for acceleration of the Proximal Gradient Descent algorithm.                    |
| [`deepinv.optim.ADMM`](https://deepinv.org/api/stubs/deepinv.optim.ADMM.html.md#deepinv.optim.ADMM)                                             | ADMM module for solving the problem                                                          |
| [`deepinv.optim.DRS`](https://deepinv.org/api/stubs/deepinv.optim.DRS.html.md#deepinv.optim.DRS)                                               | DRS module for solving the problem                                                           |
| [`deepinv.optim.HQS`](https://deepinv.org/api/stubs/deepinv.optim.HQS.html.md#deepinv.optim.HQS)                                               | Half-Quadratic Splitting (HQS) module for solving the problem                                |
| [`deepinv.optim.MD`](https://deepinv.org/api/stubs/deepinv.optim.MD.html.md#deepinv.optim.MD)                                                 | Mirror Descent (MD) or Bregman variant of the Gradient Descent algorithm.                    |
| [`deepinv.optim.PMD`](https://deepinv.org/api/stubs/deepinv.optim.PMD.html.md#deepinv.optim.PMD)                                               | Proximal Mirror Descent (PMD) or Bregman variant of the Proximal Gradient Descent algorithm. |
| [`deepinv.optim.PDCP`](https://deepinv.org/api/stubs/deepinv.optim.PDCP.html.md#deepinv.optim.PDCP)                                             | Primal Dual Chambolle-Pock optimization module.                                              |
| [`deepinv.optim.SIRT`](https://deepinv.org/api/stubs/deepinv.optim.SIRT.html.md#deepinv.optim.SIRT)                                             | Simultaneous Iterative Reconstruction Technique (SIRT) optimization module.                  |
| [`deepinv.optim.MLEM`](https://deepinv.org/api/stubs/deepinv.optim.MLEM.html.md#deepinv.optim.MLEM)                                             | Maximum Likelihood Expectation Maximization (MLEM) algorithm for Poisson inverse problems.   |
| [`deepinv.optim.OSEM`](https://deepinv.org/api/stubs/deepinv.optim.OSEM.html.md#deepinv.optim.OSEM)                                             | Ordered-Subsets Expectation-Maximization (OSEM) algorithm for Poisson inverse problems.      |

## Potentials

**User Guide:** refer to [Potentials](https://deepinv.org/user_guide/reconstruction/optimization.html.md#potentials) for more information.

| [`deepinv.optim.Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential)   | Base class for a potential $h : \xset \to \mathbb{R}$ to be used in an optimization problem.   |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|

## Data Fidelity

**User Guide:** refer to [Data Fidelity](https://deepinv.org/user_guide/reconstruction/optimization.html.md#data-fidelity) for more information.

| [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity)                             | Base class for the data fidelity term $\distance{A(x)}{y}$ where $A$ is the forward operator, $x\in\xset$ is a variable and $y\in\yset$ is the data, and where $d$ is a distance function, from the class [`deepinv.optim.Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance).   |
|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.optim.StackedPhysicsDataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.StackedPhysicsDataFidelity.html.md#deepinv.optim.StackedPhysicsDataFidelity) | Stacked data fidelity term $\datafid{x}{y} = \sum_i d_i(A_i(x),y_i)$.                                                                                                                                                                                                                                       |
| [`deepinv.optim.L1`](https://deepinv.org/api/stubs/deepinv.optim.L1.html.md#deepinv.optim.L1)                                                 | $\ell_1$ data fidelity term.                                                                                                                                                                                                                                                                                |
| [`deepinv.optim.L2`](https://deepinv.org/api/stubs/deepinv.optim.L2.html.md#deepinv.optim.L2)                                                 | Implementation of the data-fidelity as the normalized $\ell_2$ norm                                                                                                                                                                                                                                         |
| [`deepinv.optim.IndicatorL2`](https://deepinv.org/api/stubs/deepinv.optim.IndicatorL2.html.md#deepinv.optim.IndicatorL2)                               | Data-fidelity as the indicator of $\ell_2$ ball with radius $r$.                                                                                                                                                                                                                                            |
| [`deepinv.optim.PoissonLikelihood`](https://deepinv.org/api/stubs/deepinv.optim.PoissonLikelihood.html.md#deepinv.optim.PoissonLikelihood)                   | Poisson negative log-likelihood.                                                                                                                                                                                                                                                                            |
| [`deepinv.optim.LogPoissonLikelihood`](https://deepinv.org/api/stubs/deepinv.optim.LogPoissonLikelihood.html.md#deepinv.optim.LogPoissonLikelihood)             | Log-Poisson negative log-likelihood.                                                                                                                                                                                                                                                                        |
| [`deepinv.optim.AmplitudeLoss`](https://deepinv.org/api/stubs/deepinv.optim.AmplitudeLoss.html.md#deepinv.optim.AmplitudeLoss)                           | Amplitude loss as the data fidelity term for [`deepinv.physics.PhaseRetrieval()`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.html.md#deepinv.physics.PhaseRetrieval) reconstrunction.                                                                                                                              |
| [`deepinv.optim.ZeroFidelity`](https://deepinv.org/api/stubs/deepinv.optim.ZeroFidelity.html.md#deepinv.optim.ZeroFidelity)                             | Zero data fidelity term $\datafid{x}{y} = 0$.                                                                                                                                                                                                                                                               |
| [`deepinv.optim.ItohFidelity`](https://deepinv.org/api/stubs/deepinv.optim.ItohFidelity.html.md#deepinv.optim.ItohFidelity)                             | Itoh data-fidelity term for spatial unwrapping problems.                                                                                                                                                                                                                                                    |

## Priors

**User Guide:** refer to [Priors](https://deepinv.org/user_guide/reconstruction/optimization.html.md#priors) for more information.

| [`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)                                     | Prior term $\reg{x}$.                                                                          |
|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [`deepinv.optim.PnP`](https://deepinv.org/api/stubs/deepinv.optim.PnP.html.md#deepinv.optim.PnP)                                         | Plug-and-play prior $\operatorname{prox}_{\gamma \regname}(x) = \operatorname{D}_{\sigma}(x)$. |
| [`deepinv.optim.RED`](https://deepinv.org/api/stubs/deepinv.optim.RED.html.md#deepinv.optim.RED)                                         | Regularization-by-Denoising (RED) prior $\nabla \reg{x} = x - \operatorname{D}_{\sigma}(x)$.   |
| [`deepinv.optim.ScorePrior`](https://deepinv.org/api/stubs/deepinv.optim.ScorePrior.html.md#deepinv.optim.ScorePrior)                           | Score via MMSE denoiser $\nabla \reg{x}=\left(x-\operatorname{D}_{\sigma}(x)\right)/\sigma^2$. |
| [`deepinv.optim.ZeroPrior`](https://deepinv.org/api/stubs/deepinv.optim.ZeroPrior.html.md#deepinv.optim.ZeroPrior)                             | Zero prior $\reg{x} = 0$.                                                                      |
| [`deepinv.optim.Tikhonov`](https://deepinv.org/api/stubs/deepinv.optim.Tikhonov.html.md#deepinv.optim.Tikhonov)                               | Tikhonov regularizer $\reg{x} = \frac{1}{2}\| x \|_2^2$.                                       |
| [`deepinv.optim.L1Prior`](https://deepinv.org/api/stubs/deepinv.optim.L1Prior.html.md#deepinv.optim.L1Prior)                                 | $\ell_1$ prior $\reg{x} = \| x \|_1$.                                                          |
| [`deepinv.optim.WaveletPrior`](https://deepinv.org/api/stubs/deepinv.optim.WaveletPrior.html.md#deepinv.optim.WaveletPrior)                       | Wavelet prior $\reg{x} = \|\Psi x\|_{p}$.                                                      |
| [`deepinv.optim.TVPrior`](https://deepinv.org/api/stubs/deepinv.optim.TVPrior.html.md#deepinv.optim.TVPrior)                                 | Total variation (TV) prior $\reg{x} = \| D x \|_{1,2}$.                                        |
| [`deepinv.optim.TVL1Prior`](https://deepinv.org/api/stubs/deepinv.optim.TVL1Prior.html.md#deepinv.optim.TVL1Prior)                             | Total Variation (TV) prior with an L1 norm.                                                    |
| [`deepinv.optim.PatchPrior`](https://deepinv.org/api/stubs/deepinv.optim.PatchPrior.html.md#deepinv.optim.PatchPrior)                           | Patch prior $g(x) = \sum_i h(P_i x)$ for some prior $h(x)$ on the space of patches.            |
| [`deepinv.optim.L12Prior`](https://deepinv.org/api/stubs/deepinv.optim.L12Prior.html.md#deepinv.optim.L12Prior)                               | $\ell_{1,2}$ prior $\reg{x} = \sum_i\| x_i \|_2$.                                              |
| [`deepinv.optim.PatchNR`](https://deepinv.org/api/stubs/deepinv.optim.PatchNR.html.md#deepinv.optim.PatchNR)                                 | Patch prior via normalizing flows.                                                             |
| [`deepinv.optim.prior.NormalizingFlow`](https://deepinv.org/api/stubs/deepinv.optim.prior.NormalizingFlow.html.md#deepinv.optim.prior.NormalizingFlow)     | Sequential normalizing flow built from GLOW-style affine coupling blocks.                      |
| [`deepinv.optim.prior.GLOWCouplingBlock`](https://deepinv.org/api/stubs/deepinv.optim.prior.GLOWCouplingBlock.html.md#deepinv.optim.prior.GLOWCouplingBlock) | GLOW-style affine coupling block.                                                              |

## Predefined models

**User Guide:** refer to [Predefined Iterative Algorithms](https://deepinv.org/user_guide/reconstruction/iterative.html.md#predefined-iterative) for more information.

| [`deepinv.optim.DPIR`](https://deepinv.org/api/stubs/deepinv.optim.DPIR.html.md#deepinv.optim.DPIR)   | Deep Plug-and-Play (DPIR) algorithm for image restoration.   |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [`deepinv.optim.EPLL`](https://deepinv.org/api/stubs/deepinv.optim.EPLL.html.md#deepinv.optim.EPLL)   | Expected Patch Log Likelihood reconstruction method.         |

## Bregman

**User Guide:** refer to [Bregman](https://deepinv.org/user_guide/reconstruction/optimization.html.md#bregman) for more information.

| [`deepinv.optim.Bregman`](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)           | Module for the Bregman framework with convex Bregman potential $\phi$.                      |
|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`deepinv.optim.BregmanL2`](https://deepinv.org/api/stubs/deepinv.optim.BregmanL2.html.md#deepinv.optim.BregmanL2)       | Module for the L2 norm as Bregman potential $\phi(x) = \frac{1}{2} \|x\|_2^2$.              |
| [`deepinv.optim.BurgEntropy`](https://deepinv.org/api/stubs/deepinv.optim.BurgEntropy.html.md#deepinv.optim.BurgEntropy)   | Module for the using Burg's entropy as Bregman potential $\phi(x) = - \sum_i \log x_i$.     |
| [`deepinv.optim.NegEntropy`](https://deepinv.org/api/stubs/deepinv.optim.NegEntropy.html.md#deepinv.optim.NegEntropy)     | Module for the using negative entropy as Bregman potential $\phi(x) = \sum_i x_i \log x_i$. |
| [`deepinv.optim.Bregman_ICNN`](https://deepinv.org/api/stubs/deepinv.optim.Bregman_ICNN.html.md#deepinv.optim.Bregman_ICNN) | Module for the using a deep ICNN as Bregman potential.                                      |

## Distance

**User Guide:** refer to [Potentials](https://deepinv.org/user_guide/reconstruction/optimization.html.md#potentials) for more information.

| [`deepinv.optim.Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance)                                         | Distance $\distance{x}{y}$.                                                                                                                                  |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.optim.L2Distance`](https://deepinv.org/api/stubs/deepinv.optim.L2Distance.html.md#deepinv.optim.L2Distance)                                     | Implementation of $\distancename$ as the normalized $\ell_2$ norm                                                                                            |
| [`deepinv.optim.IndicatorL2Distance`](https://deepinv.org/api/stubs/deepinv.optim.IndicatorL2Distance.html.md#deepinv.optim.IndicatorL2Distance)                   | Indicator of $\ell_2$ ball with radius $r$.                                                                                                                  |
| [`deepinv.optim.PoissonLikelihoodDistance`](https://deepinv.org/api/stubs/deepinv.optim.PoissonLikelihoodDistance.html.md#deepinv.optim.PoissonLikelihoodDistance)       | (Negative) Log-likelihood of the Poisson distribution.                                                                                                       |
| [`deepinv.optim.L1Distance`](https://deepinv.org/api/stubs/deepinv.optim.L1Distance.html.md#deepinv.optim.L1Distance)                                     | $\ell_1$ distance                                                                                                                                            |
| [`deepinv.optim.AmplitudeLossDistance`](https://deepinv.org/api/stubs/deepinv.optim.AmplitudeLossDistance.html.md#deepinv.optim.AmplitudeLossDistance)               | Amplitude loss for [`deepinv.physics.PhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.html.md#deepinv.physics.PhaseRetrieval) reconstruction, defined as |
| [`deepinv.optim.LogPoissonLikelihoodDistance`](https://deepinv.org/api/stubs/deepinv.optim.LogPoissonLikelihoodDistance.html.md#deepinv.optim.LogPoissonLikelihoodDistance) | Log-Poisson negative log-likelihood.                                                                                                                         |
| [`deepinv.optim.ZeroDistance`](https://deepinv.org/api/stubs/deepinv.optim.ZeroDistance.html.md#deepinv.optim.ZeroDistance)                                 | Zero distance $\distance{z}{y} = 0$.                                                                                                                         |

## Iterators

**User Guide:** refer to [Predefined Algorithms](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim-iterators) for more information.

| [`deepinv.optim.FixedPoint`](https://deepinv.org/api/stubs/deepinv.optim.FixedPoint.html.md#deepinv.optim.FixedPoint)                                         | Fixed-point iterations module.                                                                                                                           |
|--------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.optim.OptimIterator`](https://deepinv.org/api/stubs/deepinv.optim.OptimIterator.html.md#deepinv.optim.OptimIterator)                                   | Base class for optimization iterators.                                                                                                                   |
| [`deepinv.optim.optim_iterators.fStep`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.fStep.html.md#deepinv.optim.optim_iterators.fStep)                   | Module for the single iteration steps on the data-fidelity term $f$.                                                                                     |
| [`deepinv.optim.optim_iterators.gStep`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.gStep.html.md#deepinv.optim.optim_iterators.gStep)                   | Module for the single iteration steps on the prior term $\lambda \regname$.                                                                              |
| [`deepinv.optim.optim_iterators.GDIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.GDIteration.html.md#deepinv.optim.optim_iterators.GDIteration)       | Iterator for Gradient Descent.                                                                                                                           |
| [`deepinv.optim.optim_iterators.PGDIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.PGDIteration.html.md#deepinv.optim.optim_iterators.PGDIteration)     | Iterator for proximal gradient descent.                                                                                                                  |
| [`deepinv.optim.optim_iterators.FISTAIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.FISTAIteration.html.md#deepinv.optim.optim_iterators.FISTAIteration) | Iterator for fast iterative soft-thresholding (FISTA).                                                                                                   |
| [`deepinv.optim.optim_iterators.CPIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.CPIteration.html.md#deepinv.optim.optim_iterators.CPIteration)       | Iterator for Chambolle-Pock.                                                                                                                             |
| [`deepinv.optim.optim_iterators.ADMMIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.ADMMIteration.html.md#deepinv.optim.optim_iterators.ADMMIteration)   | Iterator for alternating direction method of multipliers.                                                                                                |
| [`deepinv.optim.optim_iterators.DRSIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.DRSIteration.html.md#deepinv.optim.optim_iterators.DRSIteration)     | Iterator for Douglas-Rachford Splitting.                                                                                                                 |
| [`deepinv.optim.optim_iterators.HQSIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.HQSIteration.html.md#deepinv.optim.optim_iterators.HQSIteration)     | Single iteration of half-quadratic splitting.                                                                                                            |
| [`deepinv.optim.optim_iterators.MDIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.MDIteration.html.md#deepinv.optim.optim_iterators.MDIteration)       | Iterator for Mirror Descent.                                                                                                                             |
| [`deepinv.optim.optim_iterators.PMDIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.PMDIteration.html.md#deepinv.optim.optim_iterators.PMDIteration)     | Iterator for Proximal Mirror Descent (PMD).                                                                                                              |
| [`deepinv.optim.optim_iterators.SMIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.SMIteration.html.md#deepinv.optim.optim_iterators.SMIteration)       | Iterator for Spectral Methods for [`deepinv.physics.PhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.html.md#deepinv.physics.PhaseRetrieval).        |
| [`deepinv.optim.optim_iterators.MLEMIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.MLEMIteration.html.md#deepinv.optim.optim_iterators.MLEMIteration)   | Iterator for the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm for Poisson inverse problems.                                              |
| [`deepinv.optim.optim_iterators.OSEMIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.OSEMIteration.html.md#deepinv.optim.optim_iterators.OSEMIteration)   | Performs a single iteration of the OSEM algorithm, which is a classic baseline reconstruction method for inverse problems with Poisson noise statistics. |
| [`deepinv.optim.optim_iterators.SIRTIteration`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.SIRTIteration.html.md#deepinv.optim.optim_iterators.SIRTIteration)   | Iterator for the Simultaneous Iterative Reconstruction Technique (SIRT) algorithm.                                                                       |

## Linear Solvers

**User Guide:** refer to [Pseudoinverse](https://deepinv.org/user_guide/reconstruction/least-squares.html.md#least-squares) for more information.

| [`deepinv.optim.linear.least_squares`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares)                                     | Solves $\min_x \|Ax-y\|^2 + \frac{1}{\gamma}\|x-z\|^2$ using the specified solver.         |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`deepinv.optim.linear.least_squares_implicit_backward`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares_implicit_backward.html.md#deepinv.optim.linear.least_squares_implicit_backward) | Least squares solver with O(1) memory backward propagation using implicit differentiation. |
| [`deepinv.optim.linear.lsqr`](https://deepinv.org/api/stubs/deepinv.optim.linear.lsqr.html.md#deepinv.optim.linear.lsqr)                                                       | LSQR algorithm for solving linear systems.                                                 |
| [`deepinv.optim.linear.bicgstab`](https://deepinv.org/api/stubs/deepinv.optim.linear.bicgstab.html.md#deepinv.optim.linear.bicgstab)                                               | Biconjugate gradient stabilized algorithm.                                                 |
| [`deepinv.optim.linear.minres`](https://deepinv.org/api/stubs/deepinv.optim.linear.minres.html.md#deepinv.optim.linear.minres)                                                   | Minimal Residual Method for solving symmetric equations.                                   |
| [`deepinv.optim.linear.conjugate_gradient`](https://deepinv.org/api/stubs/deepinv.optim.linear.conjugate_gradient.html.md#deepinv.optim.linear.conjugate_gradient)                           | Standard conjugate gradient algorithm.                                                     |

## Utils

**User Guide:** refer to [Utils](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim-utils) for more information.

| [`deepinv.optim.utils.gradient_descent`](https://deepinv.org/api/stubs/deepinv.optim.utils.gradient_descent.html.md#deepinv.optim.utils.gradient_descent)                             | Standard gradient descent algorithm\`.                                                                                       |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.optim.phase_retrieval.correct_global_phase`](https://deepinv.org/api/stubs/deepinv.optim.phase_retrieval.correct_global_phase.html.md#deepinv.optim.phase_retrieval.correct_global_phase) | Corrects the global phase shift (and optionally magnitude scaling) of reconstructed complex signals to match the references. |
| [`deepinv.optim.phase_retrieval.spectral_methods`](https://deepinv.org/api/stubs/deepinv.optim.phase_retrieval.spectral_methods.html.md#deepinv.optim.phase_retrieval.spectral_methods)         | Utility function for spectral methods.                                                                                       |

| [`deepinv.optim.utils.GaussianMixtureModel`](https://deepinv.org/api/stubs/deepinv.optim.utils.GaussianMixtureModel.html.md#deepinv.optim.utils.GaussianMixtureModel)   | Gaussian mixture model including parameter estimation.   |
|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
