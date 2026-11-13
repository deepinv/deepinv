<a id="loss"></a>

# Training Losses

This module contains popular training losses for supervised and self-supervised learning,
which are especially designed for inverse problems.

## Introduction

All losses inherit from the base class [`deepinv.loss.Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss), which is a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

```pycon
>>> import torch
>>> import deepinv as dinv
>>> loss = dinv.loss.SureGaussianLoss(.1)
>>> physics = dinv.physics.Denoising()
>>> x = torch.ones(1, 3, 16, 16)
>>> y = physics(x)
>>> model = dinv.models.DnCNN()
>>> x_net = model(y)
>>> l = loss(x_net=x_net, y=y, physics=physics, model=model) # self-supervised loss, doesn't require ground truth x
```

<a id="supervised-losses"></a>

## Supervised Learning

Use a dataset of pairs of signals and measurements (and possibly information about the forward operator),
i.e., they can be written as $\mathcal{L}(x,\inverse{y})$.
The main loss function is [`deepinv.loss.SupLoss`](https://deepinv.org/api/stubs/deepinv.loss.SupLoss.html.md#deepinv.loss.SupLoss) which can use any [distortion metric](https://deepinv.org/user_guide/training/metric.html.md#metric).

<a id="self-supervised-losses"></a>

## Self-Supervised Learning

Use a dataset of measurement data alone (and possibly information about the forward operator),
i.e., they can be written as $\mathcal{L}(y,\inverse{y})$ and take into account information
about the forward measurement process.

Self-supervised losses can be roughly classified according to whether they are
designed to take care of the noise in the measurements, or take care of the ill-posedness
of the forward operator (e.g., incomplete operators with less measurements than pixels in the image)

#### Denoising Losses

| Loss                                                                                                           | Assumptions on Noise                          | Compatible with general forward operators   |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------|---------------------------------------------|
| [`deepinv.loss.MCLoss`](https://deepinv.org/api/stubs/deepinv.loss.MCLoss.html.md#deepinv.loss.MCLoss)                       | Small or no noise.                            | Yes                                         |
| [`deepinv.loss.Neighbor2Neighbor`](https://deepinv.org/api/stubs/deepinv.loss.Neighbor2Neighbor.html.md#deepinv.loss.Neighbor2Neighbor) | Independent noise across pixels.              | No                                          |
| [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss)         | Independent noise across measurements.        | Yes                                         |
| [`deepinv.loss.SureGaussianLoss`](https://deepinv.org/api/stubs/deepinv.loss.SureGaussianLoss.html.md#deepinv.loss.SureGaussianLoss)   | Gaussian noise                                | Yes                                         |
| [`deepinv.loss.SurePoissonLoss`](https://deepinv.org/api/stubs/deepinv.loss.SurePoissonLoss.html.md#deepinv.loss.SurePoissonLoss)     | Poisson noise                                 | Yes                                         |
| [`deepinv.loss.SurePGLoss`](https://deepinv.org/api/stubs/deepinv.loss.SurePGLoss.html.md#deepinv.loss.SurePGLoss)               | Poisson-Gaussian noise                        | Yes                                         |
| [`deepinv.loss.R2RLoss`](https://deepinv.org/api/stubs/deepinv.loss.R2RLoss.html.md#deepinv.loss.R2RLoss)                     | Poisson, Gaussian or Gamma noise              | Yes                                         |
| [`deepinv.loss.ScoreLoss`](https://deepinv.org/api/stubs/deepinv.loss.ScoreLoss.html.md#deepinv.loss.ScoreLoss)                 | Poisson, Gaussian or Gamma noise              | No                                          |
| [`deepinv.loss.SureGaussianLoss`](https://deepinv.org/api/stubs/deepinv.loss.SureGaussianLoss.html.md#deepinv.loss.SureGaussianLoss)   | Gaussian noise and multiple forward operators | No                                          |

In order to learn from incomplete data, you can either:

1. Use multiple operators (e.g., different masking patterns or performing further measurement splitting)
2. Use a single operator and leverage invariance to transformations (e.g., rotations, translations) using Equivariant Imaging.

#### Other self-supervised losses

| Loss                                                                                                                         | Assumptions                                                                                                                |
|------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.loss.EILoss`](https://deepinv.org/api/stubs/deepinv.loss.EILoss.html.md#deepinv.loss.EILoss)                                     | Assumes invariance of the signal distribution to transformations.<br/><br/><br/>(i.e. Equivariant Imaging)<br/><br/>       |
| [`deepinv.loss.EquivariantSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.EquivariantSplittingLoss.html.md#deepinv.loss.EquivariantSplittingLoss) | Same as [`EILoss`](https://deepinv.org/api/stubs/deepinv.loss.EILoss.html.md#deepinv.loss.EILoss)                                        |
| [`deepinv.loss.MOILoss`](https://deepinv.org/api/stubs/deepinv.loss.MOILoss.html.md#deepinv.loss.MOILoss)                                   | Assumes measurements observed through multiple operators.<br/><br/><br/>(i.e. Multi-Operator Imaging)<br/><br/>            |
| [`deepinv.loss.MOEILoss`](https://deepinv.org/api/stubs/deepinv.loss.MOEILoss.html.md#deepinv.loss.MOEILoss)                                 | Assumes measurements observed through multiple operators<br/><br/><br/>and invariance of the signal distribution<br/><br/> |
| [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss)                       | Assumes masks observe whole distribution.<br/><br/><br/>(i.e. measurement splitting or data undersampling)<br/><br/>       |
| [`deepinv.loss.TVLoss`](https://deepinv.org/api/stubs/deepinv.loss.TVLoss.html.md#deepinv.loss.TVLoss)                                     | Assumes images have piecewise smooth regions; based on Total Variation regularization                                      |
| [`deepinv.loss.AugmentConsistencyLoss`](https://deepinv.org/api/stubs/deepinv.loss.AugmentConsistencyLoss.html.md#deepinv.loss.AugmentConsistencyLoss)     | Assumes consistency to data augmentations.                                                                                 |
| [`deepinv.loss.ReducedResolutionLoss`](https://deepinv.org/api/stubs/deepinv.loss.ReducedResolutionLoss.html.md#deepinv.loss.ReducedResolutionLoss)       | Assumes invariance to degradation                                                                                          |

#### TIP
Splitting losses such as [`SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss)
can also be used to train the network from incomplete measurements of **multiple** forward operators.

<a id="mri-losses"></a>

### Specialized self-supervised losses for MRI

Several specialized losses are available for MRI reconstruction, particularly self-supervised losses:

#### NOTE
These losses are specialized versions of [`SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss) and [`Recorrupted2Recorrupted`](https://deepinv.org/api/stubs/deepinv.loss.R2RLoss.html.md#deepinv.loss.R2RLoss) to accelerated MRI problems.

#### MRI specialized losses

| Loss                                                                                                                           | Description                                                |
|--------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`deepinv.loss.mri.WeightedSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.WeightedSplittingLoss.html.md#deepinv.loss.mri.WeightedSplittingLoss) | Splitting loss for MRI with K-weighting                    |
| [`deepinv.loss.mri.RobustSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.RobustSplittingLoss.html.md#deepinv.loss.mri.RobustSplittingLoss)     | Splitting loss for noisy MRI with additional Noisier2Noise |
| [`deepinv.loss.mri.Phase2PhaseLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.Phase2PhaseLoss.html.md#deepinv.loss.mri.Phase2PhaseLoss)             | Splitting loss across time dimension for dynamic MRI       |
| [`deepinv.loss.mri.Artifact2ArtifactLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.Artifact2ArtifactLoss.html.md#deepinv.loss.mri.Artifact2ArtifactLoss) | Splitting loss across time dimension for sequential MRI    |
| [`deepinv.loss.mri.ENSURELoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.ENSURELoss.html.md#deepinv.loss.mri.ENSURELoss)                       | Gaussian SURE but for rank-deficient multiple operators.   |

<a id="adversarial-losses"></a>

## Adversarial Learning

Adversarial losses train a generator network by jointly training with an additional discriminator network in a minimax game.
These can be adapted to various flavours of GAN, e.g. WGAN, LSGAN.
We implement various popular (supervised and unsupervised) adversarial training frameworks below.
See [Adversarial Networks](https://deepinv.org/user_guide/reconstruction/adversarial.html.md#adversarial) for more details, and see [Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md#sphx-glr-auto-examples-adversarial-learning-demo-gan-imaging-py) for examples.
The base class for generators is [`deepinv.loss.adversarial.GeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.GeneratorLoss.html.md#deepinv.loss.adversarial.GeneratorLoss)
and for discriminators is [`deepinv.loss.adversarial.DiscriminatorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.DiscriminatorLoss.html.md#deepinv.loss.adversarial.DiscriminatorLoss).

#### Adversarial Losses Overview

| Generator Loss                                                                                                                        | Discriminator Loss                                                                                                                            | Description                                     |
|---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| [`SupAdversarialGeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.SupAdversarialGeneratorLoss.html.md#deepinv.loss.adversarial.SupAdversarialGeneratorLoss)     | [`SupAdversarialDiscriminatorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss.html.md#deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss)     | Supervised adversarial loss                     |
| [`UnsupAdversarialGeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss.html.md#deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss) | [`UnsupAdversarialDiscriminatorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss.html.md#deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss) | Unsupervised adversarial loss                   |
| [`UAIRGeneratorLoss`](https://deepinv.org/api/stubs/deepinv.loss.adversarial.UAIRGeneratorLoss.html.md#deepinv.loss.adversarial.UAIRGeneratorLoss)                         |                                                                                                                                               | Unsupervised reconstruction & adversarial loss. |

<a id="regularization-losses"></a>

## Network Regularization

These losses can be used to regularize the learned function, e.g., controlling its Lipschitz constant.

#### Network Regularization Losses Overview

| Loss                                                                                                                       | Description                                   |
|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| [`deepinv.loss.JacobianSpectralNorm`](https://deepinv.org/api/stubs/deepinv.loss.JacobianSpectralNorm.html.md#deepinv.loss.JacobianSpectralNorm)       | Controls spectral norm of the Jacobian matrix |
| [`deepinv.loss.FNEJacobianSpectralNorm`](https://deepinv.org/api/stubs/deepinv.loss.FNEJacobianSpectralNorm.html.md#deepinv.loss.FNEJacobianSpectralNorm) | Promotes a firmly non-expansive network.      |

<a id="loss-schedulers"></a>

## Loss schedulers

Loss schedulers can be used to control which losses are used when during more advanced training.
The base class is [`deepinv.loss.BaseLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.BaseLossScheduler.html.md#deepinv.loss.BaseLossScheduler).

#### Schedulers Overview

| Loss                                                                                                                                   | Description                                  |
|----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| [`deepinv.loss.RandomLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.RandomLossScheduler.html.md#deepinv.loss.RandomLossScheduler)                     | Schedule losses at random.                   |
| [`deepinv.loss.InterleavedLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.InterleavedLossScheduler.html.md#deepinv.loss.InterleavedLossScheduler)           | Schedule losses sequentially one-by-one.     |
| [`deepinv.loss.StepLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.StepLossScheduler.html.md#deepinv.loss.StepLossScheduler)                         | Activate losses at specified epoch.          |
| [`deepinv.loss.InterleavedEpochLossScheduler`](https://deepinv.org/api/stubs/deepinv.loss.InterleavedEpochLossScheduler.html.md#deepinv.loss.InterleavedEpochLossScheduler) | Schedule losses sequentially epoch-by-epoch. |
