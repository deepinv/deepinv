.. _loss:

Loss
====

This package contains popular training losses for supervised and self-supervised learning,
which are especially designed for inverse problems.

Introduction
--------------------
All losses inherit from the base class :meth:`deepinv.loss.Loss`, which is a meth:`torch.nn.Module`.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.Loss


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

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.SupLoss


Self-Supervised Learning
------------------------
Use a dataset of measurement data alone (and possibly information about the forward operator),
i.e., they can be written as :math:`\mathcal{L}(y,\inverse{y})` and take into account information
about the forward measurement process.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.MCLoss
    deepinv.loss.EILoss
    deepinv.loss.MOILoss
    deepinv.loss.MOEILoss
    deepinv.loss.Neighbor2Neighbor
    deepinv.loss.SplittingLoss
    deepinv.loss.Phase2PhaseLoss
    deepinv.loss.Artifact2ArtifactLoss
    deepinv.loss.SureGaussianLoss
    deepinv.loss.SurePoissonLoss
    deepinv.loss.SurePGLoss
    deepinv.loss.TVLoss
    deepinv.loss.R2RLoss
    deepinv.loss.ScoreLoss


.. _adversarial-losses:
Adversarial Learning
--------------------
Adversarial losses train a generator network by jointly training with an additional discriminator network in a minimax game. 
We implement various popular (supervised and unsupervised) adversarial training frameworks below. These can be adapted to various flavours of GAN, e.g. WGAN, LSGAN. Generator and discriminator networks are provided in :ref:`adversarial models <adversarial-networks>`.
Training is implemented using :class:`deepinv.training.AdversarialTrainer` which overrides the standard :class:`deepinv.Trainer`. See :ref:`sphx_glr_auto_examples_adversarial-learning_demo_gan_imaging.py` for usage.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.adversarial.DiscriminatorMetric
    deepinv.loss.adversarial.GeneratorLoss
    deepinv.loss.adversarial.DiscriminatorLoss
    deepinv.loss.adversarial.SupAdversarialGeneratorLoss
    deepinv.loss.adversarial.SupAdversarialDiscriminatorLoss
    deepinv.loss.adversarial.UnsupAdversarialGeneratorLoss
    deepinv.loss.adversarial.UnsupAdversarialDiscriminatorLoss
    deepinv.loss.adversarial.UAIRGeneratorLoss

Metrics
--------
Metrics are generally used to evaluate the performance of a model.

Metrics inherit from the :class:`deepinv.loss.metric.Metric` baseclass, and take either ``x_net, x``
for a full-reference metric or ``x_net`` for a no-reference metric. 

.. note::

    Metrics may also optionally take in measurements ``y``, the physics and the model.
    The arguments are the same as in :class:`deepinv.loss.Loss`.


All metrics can perform a standard set of pre and post processing, including
operating on complex numbers, normalisation and reduction. See :class:`deepinv.loss.metric.Metric` for more details.

.. note::

    By default, metrics do not reduce over the batch dimension, as the usual usage is to average the metrics over a dataset yourself.
    However, you can use the ``reduction`` argument to perform reduction, e.g. if the metric is to be used as a training loss.

All metrics can be used as training losses as well by setting ``train_loss=True``.
For example, ``MSE(train_loss=True)`` replaces :class:`torch.nn.MSELoss` and ``MAE(train_loss=True)`` replaces :class:`torch.nn.L1Loss`.

Metrics can either be used directly or as the backbone for loss functions:

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>> m = dinv.metric.SSIM()
    >>> x = torch.ones(2, 3, 16, 16) # B,C,H,W
    >>> x_hat = x + 0.01
    >>> m(x_hat, x) # Calculate metric for each image in batch
    tensor([1.0000, 1.0000])
    >>> m = dinv.metric.SSIM(reduction="sum")
    >>> m(x_hat, x) # Sum over batch
    tensor(1.9999)
    >>> l = dinv.loss.MCLoss(metric=dinv.metric.SSIM(train_loss=True, reduction="mean")) # Use SSIM for training

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

        deepinv.loss.metric.Metric
        deepinv.loss.metric.MSE
        deepinv.loss.metric.NMSE
        deepinv.loss.metric.MAE
        deepinv.loss.metric.PSNR
        deepinv.loss.metric.SSIM
        deepinv.loss.metric.L1L2
        deepinv.loss.metric.LPIPS
        deepinv.loss.metric.NIQE


Network Regularization
----------------------
These losses can be used to regularize the learned function, e.g., controlling its Lipschitz constant.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.JacobianSpectralNorm
    deepinv.loss.FNEJacobianSpectralNorm


Loss schedulers
---------------
Loss schedulers can be used to control which losses are used when during more advanced training.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.BaseLossScheduler
    deepinv.loss.RandomLossScheduler
    deepinv.loss.InterleavedLossScheduler
    deepinv.loss.InterleavedEpochLossScheduler
    deepinv.loss.StepLossScheduler


Utils
-------
A set of popular distances that can be used by the supervised and self-supervised losses.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.metric.LpNorm