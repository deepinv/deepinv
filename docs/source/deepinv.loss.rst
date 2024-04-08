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
    >>> loss(x_net=model(y), y=y, physics=physics) # self-supervised loss, doesn't require ground truth x

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
    deepinv.loss.Neighbor2Neighbor
    deepinv.loss.SplittingLoss
    deepinv.loss.SureGaussianLoss
    deepinv.loss.SurePoissonLoss
    deepinv.loss.SurePGLoss
    deepinv.loss.TVLoss
    deepinv.loss.R2RLoss

.. _adversarial-losses:
Adversarial Learning
--------------------
Adversarial losses train a generator network by jointly training with an additional discriminator network in a minimax game. 
We implement various popular adversarial training frameworks below. Generator and discriminator networks are provided in :ref:`adversarial models <adversarial-networks>`.
Training is implemented using :class:`deepinv.training.AdversarialTrainer` which overrides the standard :class:`deepinv.Trainer`. See examples for usage.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.adversarial.GeneratorLoss
    deepinv.loss.adversarial.DiscriminatorLoss
    deepinv.loss.adversarial.AmbientGANGeneratorLoss
    deepinv.loss.adversarial.AmbientGANDiscriminatorLoss
    deepinv.loss.adversarial.DeblurGANGeneratorLoss
    deepinv.loss.adversarial.DeblurGANDiscriminatorLoss
    deepinv.loss.adversarial.UAIRGeneratorLoss
    deepinv.loss.adversarial.UAIRDiscriminatorLoss

Metrics
--------
Metrics are generally used to evaluate the performance of a model. Some of them can be used as training losses as well.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

        deepinv.loss.PSNR
        deepinv.loss.SSIM
        deepinv.loss.LPIPS
        deepinv.loss.NIQE


Transforms
^^^^^^^^^^

This submodule contains different transforms which can be used for data augmentation or together with the equivariant losses.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.transform.Rotate
    deepinv.transform.Shift
    deepinv.transform.Scale

Network Regularization
----------------------
These losses can be used to regularize the learned function, e.g., controlling its Lipschitz constant.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.JacobianSpectralNorm
    deepinv.loss.FNEJacobianSpectralNorm


Utils
-------
A set of popular distances that can be used by the supervised and self-supervised losses.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.LpNorm