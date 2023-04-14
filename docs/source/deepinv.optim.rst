Optim
===============================

This package contains a collection of routines that optimize

.. math::

    \underset{x}{\arg\min} \quad \datafid{\forw{x}}{y} + \reg{x}


where the first term :math:`f:\yset\times\yset \mapsto \mathbb{R}_{+}` enforces data-fidelity
(:math:`y \approx A(x)`), the second term :math:`g:\xset\mapsto \mathbb{R}_{+}` acts as a regularization, and
:math:`A:\xset\mapsto \yset` is the forward operator (see :meth:`deepinv.physics.Physics`).


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optimizers
   deepinv.optim.utils

Data Fidelity
-------------------------------------
.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.DataFidelity
   deepinv.optim.L1
   deepinv.optim.L2
   deepinv.optim.IndicatorL2
   deepinv.optim.PoissonLikelihood


The module works using iterators... (TODO)

Iterators
-------------------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optim_iterators.ADMMIteration
   deepinv.optim.optim_iterators.PGDIteration
   deepinv.optim.optim_iterators.PDIteration
   deepinv.optim.optim_iterators.HQSIteration
   deepinv.optim.optim_iterators.DRSIteration