.. _optim:

Optim
===============================

This package contains a collection of routines that optimize

.. math::
    \begin{equation}
    \label{eq:min_prob}
    \tag{1}
    \underset{x}{\arg\min} \quad \datafid{\forw{x}}{y} + \reg{x}
    \end{equation}


where the first term :math:`f:\yset\times\yset \mapsto \mathbb{R}_{+}` enforces data-fidelity
(:math:`y \approx A(x)`), the second term :math:`g:\xset\mapsto \mathbb{R}_{+}` acts as a regularization, and
:math:`A:\xset\mapsto \yset` is the forward operator (see :meth:`deepinv.physics.Physics`).

Optimisation algorithms for minimising the problem above can be written as fixed point algorithms,
i.e. for :math:`k=1,2,...`

.. math::
    \qquad (x_{k+1}, u_{k+1}) = \operatorname{FixedPoint}(x_k, u_k, f, g, A, y, ...)

where :math:`x` is a primal variable converging to the solution of the minimisation problem, and
:math:`u` is a dual variable.
The implementation of the fixed point algorithm in :meth:`deepinv.optim`,
following standard optimisation theory, is split in two steps:

.. math::
    u_{k+1} = \operatorname{step}_f(x_k, u_k, y, A, ...)\\
    x_{k+1} = \operatorname{step}_g(x_k, u_k, y, A, ...)

where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are gradient and/or proximal steps
on :math:`f` and :math:`g`, while using additional inputs, such as :math:`A` and :math:`y`, but also stepsizes,
relaxation parameters, etc...


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.BaseOptim
   deepinv.optim.FixedPoint


.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.optim.optimbuilder


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