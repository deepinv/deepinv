.. _optim:

Optim
=====

This package contains a collection of routines that optimize

.. math::
    \begin{equation}
    \label{eq:min_prob}
    \tag{1}
    \underset{x}{\arg\min} \quad \lambda \datafid{x}{y} + \reg{x},
    \end{equation}


where the first term :math:`\datafidname:\xset\times\yset \mapsto \mathbb{R}_{+}` enforces data-fidelity, the second
term :math:`\regname:\xset\mapsto \mathbb{R}_{+}` acts as a regularization and
:math:`\lambda > 0` is a regularization parameter. More precisely, the data-fidelity term penalizes the discrepancy
between the data :math:`y` and the forward operator :math:`A` applied to the variable :math:`x`, as

.. math::
    \datafid{x}{y} = \distance{Ax}{y}

where :math:`\distance{\cdot}{\cdot}` is a distance function, and where :math:`A:\xset\mapsto \yset` is the forward
operator (see :meth:`deepinv.physics.Physics`)

Optimization algorithms for minimizing the problem above can be written as fixed point algorithms,
i.e. for :math:`k=1,2,...`

.. math::
    \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, g, A, y, ...)

where :math:`x` is a variable converging to the solution of the minimisation problem, and
:math:`z` is an additional (dual) variable that may be required in the computation of the fixed point operator.
The implementation of the fixed point algorithm in ``deepinv.optim``,
following standard optimization theory, is split in two steps:

.. math::
    z_{k+1} = \operatorname{step}_f(x_k, z_k, y, A, ...)\\
    x_{k+1} = \operatorname{step}_g(x_k, z_k, y, A, ...)

where :math:`\operatorname{step}_{\datafidname}` and :math:`\operatorname{step}_{\regname}` are gradient and/or proximal steps
on :math:`\datafidname` and :math:`\regname`, while using additional inputs, such as :math:`A` and :math:`y`, but also stepsizes,
relaxation parameters, etc...


Base Optimization Class
---------------------------

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

   deepinv.optim.optim_builder


Data Fidelity
-------------
This is the base class for the data fidelity term :math:`\distance{Ax}{y}` where :math:`A` is a linear operator,
:math:`x\in\xset` is a variable and :math:`y\in\yset` is the data, and where :math:`d` is a convex function.

This class comes with methods, such as :math:`\operatorname{prox}_{\distancename\circ A}` and
:math:`\nabla (\distancename \circ A)` (among others), on which optimization algorithms rely.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.DataFidelity
   deepinv.optim.L1
   deepinv.optim.L2
   deepinv.optim.IndicatorL2
   deepinv.optim.PoissonLikelihood


Priors
------
This is the base class for implementing prior functions :math:`\reg{x}` where :math:`x\in\xset` is a variable and
where :math:`\regname` is a function.

Similarly to the :meth:`deepinv.optim.DataFidelity` class, this class comes with methods for computing
:math:`\operatorname{prox}_{g}` and :math:`\nabla \regname`.  This base class is used to implement user-defined differentiable
priors, such as the Tikhonov regularisation, but also implicit priors. For instance, in PnP methods, the method
computing the proximity operator is overwritten by a method performing denoising.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.Prior
   deepinv.optim.PnP
   deepinv.optim.RED
   deepinv.optim.ScorePrior
   deepinv.optim.Tikhonov


Iterators
---------
An optim iterator is an object that implements a fixed point iteration for minimizing the sum of two functions
:math:`F = \lambda \datafidname + \regname` where :math:`\datafidname` is a data-fidelity term  that will be modeled by an instance of physics
and :math:`\regname` is a regularizer. The fixed point iteration takes the form

.. math::
    \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, \datafidname, \regname, A, y, ...)

where :math:`x` is a variable converging to the solution of the minimisation problem, and
:math:`z` is an additional variable that may be required in the computation of the fixed point operator.

The implementation of the fixed point algorithm in :meth:`deepinv.optim`,
following standard optimization theory, is split in two steps:

.. math::
    z_{k+1} = \operatorname{step}_{\datafidname}(x_k, z_k, y, A, ...)\\
    x_{k+1} = \operatorname{step}_{\regname}(x_k, z_k, y, A, ...)

where :math:`\operatorname{step}_{\datafidname}` and :math:`\operatorname{step}_g` are gradient and/or proximal steps
on :math:`\datafidname` and :math:`\regname`, while using additional inputs, such as :math:`A` and :math:`y`, but also stepsizes,
relaxation parameters, etc...

The fStep and gStep classes precisely implement these steps.


Generic optimizers
^^^^^^^^^^^^^^^^^^

The following files contain the base classes on which optimization algorithms rely.

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
   deepinv.optim.optim_iterators.primal_dual_CP.fStepCP
   deepinv.optim.optim_iterators.primal_dual_CP.gStepCP



