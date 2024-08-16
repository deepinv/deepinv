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
The projective transformations formulate the image transformations using the pinhole camera model, from which various transformation subgroups can be derived. See the self-supervised example for a demonstration. Note these require ``kornia`` installed.

Transforms inherit from :class:`deepinv.transform.Transform`. Transforms can also be stacked by summing them, and chained by multiplying them (i.e. product group). For example, random transforms can be used as follows:

.. doctest::

    >>> import torch
    >>> from deepinv.transform import Shift, Rotate
    >>> x = torch.rand((1, 1, 2, 2)) # Define random image (B,C,H,W)
    >>> transform = Shift() # Define random shift transform
    >>> transform(x).shape
    torch.Size([1, 1, 2, 2])
    >>> transform = Rotate() + Shift() # Stack rotate and shift transforms
    >>> transform(x).shape
    torch.Size([2, 1, 2, 2])
    >>> rotoshift = Rotate() * Shift() # Chain rotate and shift transforms
    >>> rotoshift(x).shape
    torch.Size([1, 1, 2, 2])

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.transform.Transform
    deepinv.transform.Rotate
    deepinv.transform.Shift
    deepinv.transform.Scale
    deepinv.transform.Homography
    deepinv.transform.projective.Euclidean
    deepinv.transform.projective.Similarity
    deepinv.transform.projective.Affine
    deepinv.transform.projective.PanTiltRotate

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