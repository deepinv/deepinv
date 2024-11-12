deepinv.loss
============

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.Loss


Supervised Learning
--------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.SupLoss


Self-Supervised Learning
------------------------

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

