deepinv.optim
=============

This module provides optimization utils for constructing reconstruction models based on optimization algorithms.
Please refer to the :ref:`user guide <optim>` for more details.


Base Class
----------
.. userguide:: optim

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.optim.optim_builder
   deepinv.optim.BaseOptim

Potentials
----------
.. userguide:: potentials

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.Potential

Data Fidelity
-------------
.. userguide:: data-fidelity

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
.. userguide:: priors

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
.. userguide:: predefined-iterative

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.DPIR
   deepinv.optim.EPLL


Bregman
-------
.. userguide:: bregman

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.bregman.Bregman
   deepinv.optim.bregman.BregmanL2
   deepinv.optim.bregman.BurgEntropy
   deepinv.optim.bregman.NegEntropy
   deepinv.optim.bregman.Bregman_ICNN

Iterators
---------
.. userguide:: optim-iterators

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
-----
.. userguide:: optim-utils

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.optim.utils.conjugate_gradient
    deepinv.optim.utils.gradient_descent
    deepinv.optim.utils.GaussianMixtureModel