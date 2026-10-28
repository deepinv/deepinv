# deepinv.loss

This module provides a collection of supervised and self-supervised loss functions for training reconstruction networks.
Refer to the [user guide](https://deepinv.org/user_guide/training/loss.md#loss) for more information.

## Base class

**User Guide:** refer to [Training Losses](https://deepinv.org/user_guide/training/loss.md#loss) for more information.

| [`deepinv.loss.Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.md#deepinv.loss.Loss)                             | Base class for all loss functions.           |
|------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| [`deepinv.loss.StackedPhysicsLoss`](https://deepinv.org/api/stubs/deepinv.loss.StackedPhysicsLoss.md#deepinv.loss.StackedPhysicsLoss) | Loss function for stacked physics operators. |

## Supervised Learning

**User Guide:** refer to [Supervised Learning](https://deepinv.org/user_guide/training/loss.md#supervised-losses) for more information.

| [`deepinv.loss.SupLoss`](https://deepinv.org/api/stubs/deepinv.loss.SupLoss.md#deepinv.loss.SupLoss)   | Standard supervised loss   |
|----------------------------------------------------------------------------------------------|----------------------------|

## Self-Supervised Learning

**User Guide:** refer to [Self-Supervised Learning](https://deepinv.org/user_guide/training/loss.md#self-supervised-losses) for more information.

| [`deepinv.loss.MCLoss`](https://deepinv.org/api/stubs/deepinv.loss.MCLoss.md#deepinv.loss.MCLoss)                                     | Measurement consistency loss                                |
|------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [`deepinv.loss.EILoss`](https://deepinv.org/api/stubs/deepinv.loss.EILoss.md#deepinv.loss.EILoss)                                     | Equivariant imaging self-supervised loss.                   |
| [`deepinv.loss.EquivariantSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.EquivariantSplittingLoss.md#deepinv.loss.EquivariantSplittingLoss) | Equivariant splitting loss.                                 |
| [`deepinv.loss.MOILoss`](https://deepinv.org/api/stubs/deepinv.loss.MOILoss.md#deepinv.loss.MOILoss)                                   | Multi-operator imaging loss                                 |
| [`deepinv.loss.MOEILoss`](https://deepinv.org/api/stubs/deepinv.loss.MOEILoss.md#deepinv.loss.MOEILoss)                                 | Multi-operator equivariant imaging.                         |
| [`deepinv.loss.Neighbor2Neighbor`](https://deepinv.org/api/stubs/deepinv.loss.Neighbor2Neighbor.md#deepinv.loss.Neighbor2Neighbor)               | Neighbor2Neighbor loss.                                     |
| [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.md#deepinv.loss.SplittingLoss)                       | Measurement splitting loss.                                 |
| [`deepinv.loss.SureGaussianLoss`](https://deepinv.org/api/stubs/deepinv.loss.SureGaussianLoss.md#deepinv.loss.SureGaussianLoss)                 | SURE loss for Gaussian noise                                |
| [`deepinv.loss.SurePoissonLoss`](https://deepinv.org/api/stubs/deepinv.loss.SurePoissonLoss.md#deepinv.loss.SurePoissonLoss)                   | SURE loss for Poisson noise                                 |
| [`deepinv.loss.SurePGLoss`](https://deepinv.org/api/stubs/deepinv.loss.SurePGLoss.md#deepinv.loss.SurePGLoss)                             | SURE loss for Poisson-Gaussian noise                        |
| [`deepinv.loss.TVLoss`](https://deepinv.org/api/stubs/deepinv.loss.TVLoss.md#deepinv.loss.TVLoss)                                     | Total variation loss ($\ell_2$ norm).                       |
| [`deepinv.loss.R2RLoss`](https://deepinv.org/api/stubs/deepinv.loss.R2RLoss.md#deepinv.loss.R2RLoss)                                   | Generalized Recorrupted-to-Recorrupted (GR2R) Loss          |
| [`deepinv.loss.ScoreLoss`](https://deepinv.org/api/stubs/deepinv.loss.ScoreLoss.md#deepinv.loss.ScoreLoss)                               | Learns score of distribution in the context of Noise2Score. |
| [`deepinv.loss.AugmentConsistencyLoss`](https://deepinv.org/api/stubs/deepinv.loss.AugmentConsistencyLoss.md#deepinv.loss.AugmentConsistencyLoss)     | Data augmentation consistency (DAC) loss.                   |
| [`deepinv.loss.ReducedResolutionLoss`](https://deepinv.org/api/stubs/deepinv.loss.ReducedResolutionLoss.md#deepinv.loss.ReducedResolutionLoss)       | Reduced resolution loss for blur and downsampling problems. |

### Specialized self-supervised losses for MRI

**User Guide:** refer to [Specialized self-supervised losses for MRI](https://deepinv.org/user_guide/training/loss.md#mri-losses) for more information.

| [`deepinv.loss.mri.WeightedSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.WeightedSplittingLoss.md#deepinv.loss.mri.WeightedSplittingLoss)   | K-Weighted Splitting Loss                               |
|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| [`deepinv.loss.mri.RobustSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.RobustSplittingLoss.md#deepinv.loss.mri.RobustSplittingLoss)       | Robust Weighted Splitting Loss                          |
| [`deepinv.loss.mri.Phase2PhaseLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.Phase2PhaseLoss.md#deepinv.loss.mri.Phase2PhaseLoss)               | Phase2Phase loss for dynamic data.                      |
| [`deepinv.loss.mri.Artifact2ArtifactLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.Artifact2ArtifactLoss.md#deepinv.loss.mri.Artifact2ArtifactLoss)   | Artifact2Artifact loss for dynamic data.                |
| [`deepinv.loss.mri.ENSURELoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.ENSURELoss.md#deepinv.loss.mri.ENSURELoss)                         | ENSURE loss for image reconstruction in Gaussian noise. |

## Adversarial Learning

**User Guide:** refer to [Adversarial Learning](https://deepinv.org/user_guide/training/loss.md#adversarial-losses) for more information.

| [`deepinv.loss.adversarial.DiscriminatorMetric`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.DiscriminatorMetric.md#deepinv.loss.adversarial.DiscriminatorMetric)                             | Generic GAN discriminator metric building block.             |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [`deepinv.loss.adversarial.GeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.GeneratorLoss.md#deepinv.loss.adversarial.GeneratorLoss)                                         | Base generator adversarial loss.                             |
| [`deepinv.loss.adversarial.DiscriminatorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.DiscriminatorLoss.md#deepinv.loss.adversarial.DiscriminatorLoss)                                 | Base discriminator adversarial loss.                         |
| [`deepinv.loss.adversarial.SupAdversarialGeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.SupAdversarialGeneratorLoss.md#deepinv.loss.adversarial.SupAdversarialGeneratorLoss)             | Supervised adversarial consistency loss for generator.       |
| [`deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss.md#deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss)     | Supervised adversarial consistency loss for discriminator.   |
| [`deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss.md#deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss)         | Unsupervised adversarial consistency loss for generator.     |
| [`deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss.md#deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss) | Unsupervised adversarial consistency loss for discriminator. |
| [`deepinv.loss.adversarial.UAIRGeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.UAIRGeneratorLoss.md#deepinv.loss.adversarial.UAIRGeneratorLoss)                                 | Reimplementation of UAIR generator's adversarial loss.       |

## Network Regularization

**User Guide:** refer to [Network Regularization](https://deepinv.org/user_guide/training/loss.md#regularization-losses) for more information.

| [`deepinv.loss.JacobianSpectralNorm`](https://deepinv.org/api/stubs/deepinv.loss.JacobianSpectralNorm.md#deepinv.loss.JacobianSpectralNorm)       | Computes the spectral norm of the Jacobian.                |
|----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`deepinv.loss.FNEJacobianSpectralNorm`](https://deepinv.org/api/stubs/deepinv.loss.FNEJacobianSpectralNorm.md#deepinv.loss.FNEJacobianSpectralNorm) | Computes the Firm-Nonexpansiveness Jacobian spectral norm. |

## Loss schedulers

**User Guide:** refer to [Loss schedulers](https://deepinv.org/user_guide/training/loss.md#loss-schedulers) for more information.

| [`deepinv.loss.BaseLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.BaseLossScheduler.md#deepinv.loss.BaseLossScheduler)                         | Base class for loss schedulers.              |
|----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| [`deepinv.loss.RandomLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.RandomLossScheduler.md#deepinv.loss.RandomLossScheduler)                     | Schedule losses at random.                   |
| [`deepinv.loss.InterleavedLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.InterleavedLossScheduler.md#deepinv.loss.InterleavedLossScheduler)           | Schedule losses sequentially one-by-one.     |
| [`deepinv.loss.InterleavedEpochLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.InterleavedEpochLossScheduler.md#deepinv.loss.InterleavedEpochLossScheduler) | Schedule losses sequentially epoch-by-epoch. |
| [`deepinv.loss.StepLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.StepLossScheduler.md#deepinv.loss.StepLossScheduler)                         | Activate losses at specified epoch.          |
