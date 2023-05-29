.. _optim:

Optim
=====

This package contains a collection of routines that optimize

.. math::
    \begin{equation}
    \label{eq:min_prob}
    \tag{1}
    \underset{x}{\arg\min} \quad \lambda \datafid{\forw{x}}{y} + \reg{x}
    \end{equation}


where the first term :math:`f:\yset\times\yset \mapsto \mathbb{R}_{+}` enforces data-fidelity
(:math:`y \approx A(x)`), the second term :math:`g:\xset\mapsto \mathbb{R}_{+}` acts as a regularization,
:math:`\lambda > 0` is a regularization parameter, and :math:`A:\xset\mapsto \yset` is the forward operator
(see :meth:`deepinv.physics.Physics`).

Optimisation algorithms for minimizing the problem above can be written as fixed point algorithms,
i.e. for :math:`k=1,2,...`

.. math::
    \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, g, A, y, ...)

where :math:`x` is a variable converging to the solution of the minimisation problem, and
:math:`z` is an additional variable that may be required in the computation of the fixed point operator.
The implementation of the fixed point algorithm in :meth:`deepinv.optim`,
following standard optimisation theory, is split in two steps:

.. math::
    z_{k+1} = \operatorname{step}_f(x_k, z_k, y, A, ...)\\
    x_{k+1} = \operatorname{step}_g(x_k, z_k, y, A, ...)

where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are gradient and/or proximal steps
on :math:`f` and :math:`g`, while using additional inputs, such as :math:`A` and :math:`y`, but also stepsizes,
relaxation parameters, etc...


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.BaseOptim
   deepinv.optim.FixedPoint
   deepinv.optim.AndersonAcceleration


.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.optim.optim_builder


Data Fidelity
-------------
This is the base class for the data fidelity term :math:`\datafid{Ax}{y}` where :math:`A` is a linear operator,
:math:`x\in\xset` is a variable and :math:`y\in\yset` is the data, and where :math:`f` is a convex function.

This class comes with methods, such as :math:`\operatorname{prox}_{f\circ A}` and :math:`\nabla f \circ A` (among others),
on which optimization algorithms rely.

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
---------
An optim iterator is an object that implements a fixed point iteration for minimizing the sum of two functions
:math:`F = \lambda f + g` where :math:`f` is a data-fidelity term  that will be modeled by an instance of physics
and :math:`g` is a regularizer. The fixed point iteration takes the form

.. math::
    \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, g, A, y, ...)

where :math:`x` is a variable converging to the solution of the minimisation problem, and
:math:`z` is an additional variable that may be required in the computation of the fixed point operator.

The implementation of the fixed point algorithm in :meth:`deepinv.optim`,
following standard optimisation theory, is split in two steps:

.. math::
    z_{k+1} = \operatorname{step}_f(x_k, z_k, y, A, ...)\\
    x_{k+1} = \operatorname{step}_g(x_k, z_k, y, A, ...)

where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are gradient and/or proximal steps
on :math:`f` and :math:`g`, while using additional inputs, such as :math:`A` and :math:`y`, but also stepsizes,
relaxation parameters, etc...

The fStep and gStep classes precisely implement these steps.


Generic optimizers
^^^^^^^^^^^^^^^^^^

The following files contain the base classes on which optimisation algorithms rely.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optim_iterators.OptimIterator
   deepinv.optim.optim_iterators.optim_iterator.fStep
   deepinv.optim.optim_iterators.optim_iterator.gStep


ADMM
^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optim_iterators.ADMMIteration
   deepinv.optim.optim_iterators.admm.fStepADMM
   deepinv.optim.optim_iterators.admm.gStepADMM


Douglas-Rachford Splitting
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optim_iterators.DRSIteration
   deepinv.optim.optim_iterators.drs.fStepDRS
   deepinv.optim.optim_iterators.drs.gStepDRS


Gradient Descent
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optim_iterators.PGDIteration
   deepinv.optim.optim_iterators.pgd.fStepPGD
   deepinv.optim.optim_iterators.pgd.gStepPGD


Proximal Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optim_iterators.PGDIteration
   deepinv.optim.optim_iterators.pgd.fStepPGD
   deepinv.optim.optim_iterators.pgd.gStepPGD



Half-Quadratic Splitting
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optim_iterators.HQSIteration
   deepinv.optim.optim_iterators.hqs.fStepHQS
   deepinv.optim.optim_iterators.hqs.gStepHQS



Chambolle-Pock Primal-Dual Splitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.optim_iterators.CPIteration
   deepinv.optim.optim_iterators.primal_dual.fStepCP
   deepinv.optim.optim_iterators.primal_dual.gStepCP



