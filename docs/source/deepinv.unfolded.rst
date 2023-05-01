Unfolded algorithms
===============================

This package contains a collection of routines turning optimization algorithms into unfolded architectures.
Recall that optimization algorithms aim at solving problems of the form :math:`\datafid{\forw{x}}{y} + \reg{x}`
where :math:`\datafid{\cdot}{\cdot}` is a data-fidelity term, :math:`\reg{\cdot}` is a regularization term.
The resulting fixed-point algorithms for solving these problems are of the form (see :meth:`deepinv.optim`)

.. math::
    z_{k+1} = \operatorname{step}_f(x_k, z_k, y, A, ...)\\
    x_{k+1} = \operatorname{step}_g(x_k, z_k, y, A, ...)

where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are gradient and/or proximal steps on
:math:`f` and :math:`g` respectively.

Unfolded architectures are obtained by replacing parts of these algorithms by learnable structures.
In turn, they can be trained in an end-to-end fashion to solve inverse problems.

Unfolded
--------------------------------------

.. automodule:: deepinv.unfolded.unfolded
   :members:
   :undoc-members:
   :show-inheritance:


Deep Equilibrium
--------------------------------------

.. automodule:: deepinv.unfolded.deep_equilibrium
   :members:
   :undoc-members:
   :show-inheritance:
