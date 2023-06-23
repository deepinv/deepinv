.. _unfolded:

Unfolded algorithms
===================

This package contains a collection of routines turning the optimization algorithms defined in :ref:`Optim <optim>`
into unfolded architectures.
Recall that optimization algorithms aim at solving problems of the form :math:`\datafid{\forw{x}}{y} + \reg{x}`
where :math:`\datafid{\cdot}{\cdot}` is a data-fidelity term, :math:`\reg{\cdot}` is a regularization term.
The resulting fixed-point algorithms for solving these problems are of the form (see :ref:`Optim <optim>`)

.. math::
    \begin{aligned}
    z_{k+1} &= \operatorname{step}_f(x_k, z_k, y, A, ...)\\
    x_{k+1} &= \operatorname{step}_g(x_k, z_k, y, A, ...)
    \end{aligned}

where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are gradient and/or proximal steps on
:math:`f` and :math:`g` respectively.

Unfolded architectures (sometimes called 'unrolled architectures') are obtained by replacing parts of these algorithms
by learnable structures. In turn, they can be trained in an end-to-end fashion to solve inverse problems.

Unfolded
--------
The :class:`deepinv.unfolded.Unfolded` class is a generic class for building unfolded architectures. It provides
a trainable reconstruction network using a either pre-existing optimizer (e.g., "PGD") or
an iterator defined by the user. The user can choose which parameters (e.g., prior denoiser, step size, regularization
parameter, etc.) are learnable and which are not.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.unfolded.Unfolded


Deep Equilibrium
----------------
Deep equilibrium models are a particular class of unfolded architectures where the reconstruction network is defined
implicitly as the fixed-point of an optimization algorithm, i.e.,

.. math::
    \begin{aligned}
    z &= \operatorname{step}_f(x, z, y, A, ...)\\
    x &= \operatorname{step}_g(x, z, y, A, ...)
    \end{aligned}


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.unfolded.deep_equilibrium.BaseDEQ