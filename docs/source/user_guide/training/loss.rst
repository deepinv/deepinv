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


.. _supervised-losses:

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

Self-supervised losses can be roughly classified according to whether they are
designed to take care of the noise in the measurements, or take care of the ill-posedness
of the forward operator (e.g., incomplete operators with less measurements than pixels in the image)

.. list-table:: Denoising Losses
   :header-rows: 1

   * - Loss
     - Assumptions on Noise
     - Compatible with general forward operators
   * - :class:`deepinv.loss.MCLoss`
     - Small or no noise.
     - Yes
   * - :class:`deepinv.loss.Neighbor2Neighbor`
     - Independent noise across pixels.
     - No
   * - :class:`deepinv.loss.SplittingLoss`
     - Independent noise across measurements.
     - Yes
   * - :class:`deepinv.loss.Phase2PhaseLoss`
     - TODO
     - No
   * - :class:`deepinv.loss.Artifact2ArtifactLoss`
     - TODO
     - No
   * - :class:`deepinv.loss.SureGaussianLoss`
     - Gaussian noise
     - Yes
   * - :class:`deepinv.loss.SurePoissonLoss`
     - Poisson noise
     - Yes
   * - :class:`deepinv.loss.SurePGLoss`
     - Poisson-Gaussian noise
     - Yes
   * - :class:`deepinv.loss.R2RLoss`
     - Gaussian noise
     - Yes
   * - :class:`deepinv.loss.ScoreLoss`
     - Poisson, Gaussian or Gamma noise
     - No

In order to learn from incomplete data, you can either
i) use multiple operators (e.g., different masking patterns) or
ii) use a single operator and leverage invariance to transformations (e.g., rotations, translations).

.. list-table:: Other losses
   :header-rows: 1

   * - Loss
     - Assumptions
   * - :class:`deepinv.loss.EILoss`
     - Assumes invariance of the signal distribution to transformations.
   * - :class:`deepinv.loss.MOILoss`
     - Assumes measurements observed through multiple operators.
   * - :class:`deepinv.loss.MOEILoss`
     - | Assumes measurements observed through multiple operators
       | and invariance of the signal distribution
   * - :class:`deepinv.loss.TVLoss`
     - Assumes images have piecewise smooth regions; based on Total Variation regularization


.. tip::

       Splitting losses such as :class:`SplittingLoss <deepinv.loss.SplittingLoss>`, :class:`Phase2PhaseLoss <deepinv.loss.Phase2PhaseLoss>`,
       and :class:`Artifact2ArtifactLoss <deepinv.loss.Artifact2ArtifactLoss>`
       can also be used to train the network from incomplete measurements of **multiple** forward operators.

.. _regularization-losses:

Network Regularization
----------------------
These losses can be used to regularize the learned function, e.g., controlling its Lipschitz constant.

.. list-table:: Network Regularization Losses Overview
   :header-rows: 1

   * - Loss
     - Description
   * - :class:`deepinv.loss.JacobianSpectralNorm`
     - Controls spectral norm of the Jacobian matrix
   * - :class:`deepinv.loss.FNEJacobianSpectralNorm`
     - Promotes a firmly non-expansive network.

.. _adversarial-losses:

Adversarial Learning
--------------------
Adversarial losses train a generator network by jointly training with an additional discriminator network in a minimax game.
We implement various popular (supervised and unsupervised) adversarial training frameworks below. These can be adapted to various flavours of GAN, e.g. WGAN, LSGAN. Generator and discriminator networks are provided in :ref:`adversarial models <adversarial-networks>`.
Training is implemented using :class:`deepinv.training.AdversarialTrainer` which overrides the standard :class:`deepinv.Trainer`. See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for usage.

- :class:`deepinv.loss.adversarial.GeneratorLoss`
- :class:`deepinv.loss.adversarial.DiscriminatorLoss`
- :class:`deepinv.loss.adversarial.DiscriminatorMetric`

.. list-table:: Adversarial Losses Overview
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

.. _loss-schedulers:

Loss schedulers
---------------
Loss schedulers can be used to control which losses are used when during more advanced training.
The base class is :class:`deepinv.loss.BaseLossScheduler`.


.. list-table:: Schedulers Overview
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
