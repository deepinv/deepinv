deepinv.optim
=============

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.optim.optim_builder
   deepinv.optim.BaseOptim


Data Fidelity
-------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.DataFidelity
   deepinv.optim.L1
   deepinv.optim.L2
   deepinv.optim.IndicatorL2
   deepinv.optim.PoissonLikelihood
   deepinv.optim.LogPoissonLikelihood
   deepinv.optim.AmplitudeLoss


Priors
------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.Prior
   deepinv.optim.PnP
   deepinv.optim.RED
   deepinv.optim.ScorePrior
   deepinv.optim.Tikhonov
   deepinv.optim.L1Prior
   deepinv.optim.WaveletPrior
   deepinv.optim.TVPrior
   deepinv.optim.PatchPrior
   deepinv.optim.PatchNR
   deepinv.optim.L12Prior

Predefined models
-----------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.DPIR
   deepinv.optim.EPLL


Iterators
---------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.FixedPoint
   deepinv.optim.OptimIterator
   deepinv.optim.optim_iterators.GDIteration
   deepinv.optim.optim_iterators.PGDIteration
   deepinv.optim.optim_iterators.FISTAIteration
   deepinv.optim.optim_iterators.CPIteration
   deepinv.optim.optim_iterators.ADMMIteration
   deepinv.optim.optim_iterators.DRSIteration
   deepinv.optim.optim_iterators.HQSIteration
   deepinv.optim.optim_iterators.SMIteration


Utils
-------------
We provide some useful utilities for optimization algorithms.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.optim.utils.conjugate_gradient
    deepinv.optim.utils.gradient_descent
    deepinv.optim.utils.GaussianMixtureModel