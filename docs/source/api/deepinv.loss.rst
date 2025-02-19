deepinv.loss
============

This package provides a collection of supervised and self-supervised loss functions for training reconstruction networks.
Refer to the :ref:`user guide <loss>` for more information.


Base class
-----------
.. userguide:: loss

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.Loss
    deepinv.loss.StackedPhysicsLoss


Supervised Learning
--------------------
.. userguide:: supervised-losses

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.SupLoss


Self-Supervised Learning
------------------------
.. userguide:: self-supervised-losses

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

Adversarial Learning
--------------------
.. userguide:: adversarial-losses

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

Network Regularization
----------------------
.. userguide:: regularization-losses

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.JacobianSpectralNorm
    deepinv.loss.FNEJacobianSpectralNorm


Loss schedulers
---------------
.. userguide:: loss-schedulers

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.BaseLossScheduler
    deepinv.loss.RandomLossScheduler
    deepinv.loss.InterleavedLossScheduler
    deepinv.loss.InterleavedEpochLossScheduler
    deepinv.loss.StepLossScheduler

