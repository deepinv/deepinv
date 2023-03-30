Optim
===============================

This package contains a collection of routines that optimize

.. math::

    \underset{x}{\arg\min} \quad f(y,A(x)) + g(x)


where the first term :math:`f:\mathbb{R}^{m}\times\mathbb{R}^{m} \mapsto \mathbb{R}_{+}` enforces data-fidelity
(:math:`y \approx A(x)`), the second term :math:`g:\mathbb{R}^{n}\mapsto \mathbb{R}_{+}` acts as a regularization, and
:math:`A:\mathbb{R}^{n}\mapsto \mathbb{R}^{m}` is the forward operator (see ``deepinv.physics``).


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optimizers
   deepinv.optim.data_fidelity
   deepinv.optim.utils

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