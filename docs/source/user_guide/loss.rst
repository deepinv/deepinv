.. _loss:

Training Losses
===============

This package contains popular training losses for supervised and self-supervised learning,
which are especially designed for inverse problems.

Introduction
--------------------
All losses inherit from the base class :class:`deepinv.loss.Loss`, which is a :class:`torch.nn.Module`.


.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>> loss = dinv.loss.SureGaussianLoss(.1)
    >>> physics = dinv.physics.Denoising()
    >>> x = torch.ones(1, 3, 16, 16)
    >>> y = physics(x)
    >>> model = dinv.models.DnCNN()
    >>> x_net = model(y)
    >>> l = loss(x_net=x_net, y=y, physics=physics, model=model) # self-supervised loss, doesn't require ground truth x

Supervised Learning
--------------------
Use a dataset of pairs of signals and measurements (and possibly information about the forward operator),
i.e., they can be written as :math:`\mathcal{L}(x,\inverse{y})`.
The main loss function is :class:`deepinv.loss.SupLoss` which can use any :ref:`distortion metric <metric>`.

.. _self-supervised-losses:

Self-Supervised Learning
------------------------
Use a dataset of measurement data alone (and possibly information about the forward operator),
i.e., they can be written as :math:`\mathcal{L}(y,\inverse{y})` and take into account information
about the forward measurement process.

.. list-table:: Self-Supervised Losses Overview
   :widths: 25 35
   :header-rows: 1

   * - Loss
     - Assumptions on Noise
   * - :class:`deepinv.loss.MCLoss`
     - Monte Carlo noise estimation
   * - :class:`deepinv.loss.Neighbor2Neighbor`
     - Neighboring pixels share similar information, suited for structured noise
   * - :class:`deepinv.loss.SplittingLoss`
     - Data can be split into noisy and clean components
   * - :class:`deepinv.loss.Phase2PhaseLoss`
     - Phase noise is predominant, with phase consistency assumed
   * - :class:`deepinv.loss.Artifact2ArtifactLoss`
     - Noise primarily causes consistent artifacts
   * - :class:`deepinv.loss.SureGaussianLoss`
     - Assumes Gaussian noise, based on Stein’s Unbiased Risk Estimator (SURE)
   * - :class:`deepinv.loss.SurePoissonLoss`
     - Assumes Poisson noise, based on SURE
   * - :class:`deepinv.loss.SurePGLoss`
     - Assumes Poisson-Gaussian noise, based on SURE
   * - :class:`deepinv.loss.R2RLoss`
     - Suited for paired noisy data, no clean reference required
   * - :class:`deepinv.loss.ScoreLoss`
     - Assumes score-based or noise-injected data for training


.. list-table:: Losses Overview
   :widths: 25 35
   :header-rows: 1

   * - Loss
     - Assumptions
   * - :class:`deepinv.loss.EILoss`
     - Assumes existence of an energy functional for image reconstruction
   * - :class:`deepinv.loss.MOILoss`
     - Multi-objective optimization framework; assumes multiple conflicting objectives
   * - :class:`deepinv.loss.MOEILoss`
     - Multi-objective energy minimization; assumes compatibility with multiple energy terms
   * - :class:`deepinv.loss.TVLoss`
     - Assumes images have piecewise smooth regions; based on Total Variation (TV) regularization


Network Regularization
----------------------
These losses can be used to regularize the learned function, e.g., controlling its Lipschitz constant.

.. list-table:: Network Regularization Losses Overview
   :widths: 25 45
   :header-rows: 1

   * - Loss
     - Description
   * - :class:`deepinv.loss.JacobianSpectralNorm`
     - Computes the spectral norm of the Jacobian matrix to regularize the model, helping to control sensitivity to input perturbations.
   * - :class:`deepinv.loss.FNEJacobianSpectralNorm`
     - Fast Neural Estimation of the Jacobian spectral norm; optimized for efficiency in calculating the spectral norm, suitable for large-scale models.

.. _adversarial-losses:

Adversarial Learning
--------------------
Adversarial losses train a generator network by jointly training with an additional discriminator network in a minimax game.
We implement various popular (supervised and unsupervised) adversarial training frameworks below. These can be adapted to various flavours of GAN, e.g. WGAN, LSGAN. Generator and discriminator networks are provided in :ref:`adversarial models <adversarial-networks>`.
Training is implemented using :class:`deepinv.training.AdversarialTrainer` which overrides the standard :class:`deepinv.Trainer`. See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for usage.

:class:`deepinv.loss.adversarial.GeneratorLoss`
:class:`deepinv.loss.adversarial.DiscriminatorLoss`
:class:`deepinv.loss.adversarial.DiscriminatorMetric`

.. list-table:: Adversarial Losses Overview
   :widths: 35 35
   :header-rows: 1

   * - Generator Loss
     - Discriminator Loss
     - Description
   * - :class:`deepinv.loss.adversarial.SupAdversarialGeneratorLoss`
     - :class:`deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss`
     - Supervised adversarial training
   * - :class:`deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss`
     - :class:`deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss`
     - Unsupervised adversarial training
   * - :class:`deepinv.loss.adversarial.UAIRGeneratorLoss`
     -
     - Unsupervised Adversarial Image Reconstruction loss.



Loss schedulers
---------------
Loss schedulers can be used to control which losses are used when during more advanced training.
The base class is :class:`deepinv.loss.BaseLossScheduler`.


.. list-table:: Schedulers Overview
   :widths: 25 45
   :header-rows: 1

   * - Loss
     - Description
   * - :class:`deepinv.loss.RandomLossScheduler`
     - Schedule losses at random.
   * - :class:`deepinv.loss.InterleavedLossScheduler`
     - Schedule losses sequentially one-by-one.
   * - :class:`deepinv.loss.StepLossScheduler`
     - Activate losses at specified epoch.
   * - :class:`deepinv.loss.InterleavedEpochLossScheduler`
     - Schedule losses sequentially epoch-by-epoch.
