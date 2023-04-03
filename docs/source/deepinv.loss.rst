Loss
====================

This package contains popular training losses for supervised and self-supervised learning,
which are especially designed for inverse problems.


Supervised Learning
-----------------------
Use a dataset of pairs of signals and measurements (and possibly information about the forward operator),
i.e., they can be written as :math:`\mathcal{L}(x,f(y))`.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.SupLoss


Self-Supervised Learning
---------------------------
Use a dataset of measurement data alone (and possibly information about the forward operator),
i.e., they can be written as :math:`\mathcal{L}(y,f(y))` and take into account information
about the forward measurement process.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.MCLoss
    deepinv.loss.EILoss
    deepinv.loss.MOILoss
    deepinv.loss.SplittingLoss
    deepinv.loss.SureGaussianLoss
    deepinv.loss.SurePoissonLoss
    deepinv.loss.SurePGLoss
    deepinv.loss.TVLoss


Network Regularization
---------------------------
These losses can be used to regularize the learned function, e.g., controlling its Lipschitz constant.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.JacobianSpectralNorm
    deepinv.loss.FNEJacobianSpectralNorm


Metrics
------------------------
A set of popular metrics that can be used by the supervised and self-supervised losses.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.LpNorm
    deepinv.loss.CharbonnierLoss
