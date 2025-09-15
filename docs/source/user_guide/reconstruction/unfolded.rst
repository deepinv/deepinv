.. _unfolded:

Unfolded Algorithms
===================

This module contains a collection of routines turning the optimization algorithms defined in :ref:`optimization module <optim>`
into unfolded architectures.
Recall that optimization algorithms aim at solving problems of the form :math:`\datafid{x}{y} + \reg{x}`
where :math:`\datafid{\cdot}{\cdot}` is a data-fidelity term, :math:`\reg{\cdot}` is a regularization term.
The resulting fixed-point algorithms for solving these problems are of the form (see :ref:`optimization <optim>`)

.. math::

    \begin{aligned}
    z_{k+1} &= \operatorname{step}_f(x_k, z_k, y, A, ...)\\
    x_{k+1} &= \operatorname{step}_g(x_k, z_k, y, A, ...)
    \end{aligned}

where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are gradient and/or proximal steps on
:math:`f` and :math:`g` respectively.

Unfolded architectures (sometimes called 'unrolled architectures') are obtained by replacing parts of these algorithms
by learnable modules. In turn, they can be trained in an end-to-end fashion to solve inverse problems.

Unfolded
--------
The :class:`deepinv.unfolded.unfolded_builder` class is a generic class for building unfolded architectures. It provides
a trainable reconstruction network using a either pre-existing optimizer (e.g., "PGD") or
an iterator defined by the user. The user can choose which parameters (e.g., prior denoiser, step size, regularization
parameter, etc.) are learnable and which are not.

The builder depends on the backbone class for DEQs, :class:`deepinv.unfolded.BaseUnfold`.



In the following example, we create an unfolded architecture of 5 proximal gradient steps
using a DnCNN plug-and-play prior a standard L2 data-fidelity term. The network can be trained end-to-end, and
evaluated with any forward model (e.g., denoising, deconvolution, inpainting, etc.).

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>>
    >>> # Create a trainable unfolded architecture
    >>> model = dinv.unfolded.unfolded_builder(  # doctest: +IGNORE_RESULT
    ...     iteration="PGD",
    ...     data_fidelity=dinv.optim.L2(),
    ...     prior=dinv.optim.PnP(dinv.models.DnCNN()),
    ...     params_algo={"stepsize": 1.0, "g_param": 1.0},
    ...     trainable_params=["stepsize", "g_param"]
    ... )
    >>> # Forward pass
    >>> x = torch.randn(1, 3, 16, 16)
    >>> physics = dinv.physics.Denoising()
    >>> y = physics(x)
    >>> x_hat = model(y, physics)

Memory-efficient back-propagation
-------------------------------------------
Some unfolded architectures rely on a least-squares solver to compute the proximal step w.r.t. the data-fidelity term (e.g., :class:`ADMM <deepinv.optim.optim_iterators.ADMMIteration>` or :class:`HQS <deepinv.optim.optim_iterators.HQSIteration>`). During backpropagation, a naive implementation requires storing the gradients of every intermediate step of the least squares solver, resulting in significant memory and computational costs.
We provide a memory-efficient back-propagation strategy that reduces the memory footprint during training, by computing the gradients of the least squares solver in closed-form, without storing any intermediate steps. This closed-form computation requires evaluating the least-squares solver one additional time during the gradient computation.
This is particularly useful when training deep unfolded architectures with many iterations. 
To enable this feature, set the argument ``implicit_backward_solver=True`` (default is `True`) when creating the physics. It is supported for all linear physics
(:class:`deepinv.physics.LinearPhysics`).  
Note that when setting ``implicit_backward_solver=True``, we need to use a large enough number of iterations in the physics solver to ensure convergence (as the closed-form gradients assume that we converged to the least squares minimizer), otherwise the gradient might be inaccurate.
See also :ref:`sphx_glr_auto_examples_unfolded_demo_unfolded_constant_memory.py` for more details.

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>>
    >>> # Create a trainable unfolded architecture
    >>> model = dinv.unfolded.unfolded_builder(  # doctest: +IGNORE_RESULT
    ...     iteration="HQS",
    ...     data_fidelity=dinv.optim.L2(),
    ...     prior=dinv.optim.PnP(dinv.models.DnCNN()),
    ...     params_algo={"stepsize": 1.0, "g_param": 1.0},
    ...     trainable_params=["stepsize", "g_param"]
    ... )
    >>> # Forward pass
    >>> x = torch.randn(1, 3, 16, 16)
    >>> physics = dinv.physics.Blur(filter=torch.ones(1, 1, 3, 3) / 9., implicit_backward_solver=True, max_iter=50)
    >>> y = physics(x)
    >>> x_hat = model(y, physics)


.. _deep-equilibrium:

Deep Equilibrium
----------------
Deep Equilibrium models (DEQ) are a particular class of unfolded architectures where the backward pass
is performed via Fixed-Point iterations. DEQ algorithms can virtually unroll infinitely many layers leveraging
the **implicit function theorem**. The backward pass consists in looking for solutions of the fixed-point equation

.. math::

   v = \left(\frac{\partial \operatorname{FixedPoint}(x^\star)}{\partial x^\star} \right)^{\top} v + u.


where :math:`u` is the incoming gradient from the backward pass,
and :math:`x^\star` is the equilibrium point of the forward pass.
See `this tutorial <http://implicit-layers-tutorial.org/deep_equilibrium_models/>`_ for more details.

The :class:`deepinv.unfolded.DEQ_builder` class is a generic class for building Deep Equilibrium (DEQ) architectures.


The builder depends on the backbone class for DEQs, :class:`deepinv.unfolded.BaseDEQ`.

.. _predefined-unfolded:

Predefined Unfolded Architectures
---------------------------------
We also provide some off-the-shelf unfolded network architectures,
taken from the respective literatures.

.. list-table:: Predefined unfolded architectures
   :header-rows: 1

   * - Model
     - Description
   * - :class:`deepinv.models.VarNet`
     - VarNet/E2E-VarNet MRI reconstruction models
   * - :class:`deepinv.models.MoDL`
     - MoDL MRI reconstruction model

.. _custom-unfolded-blocks:

Predefined Unfolded Blocks
--------------------------
Some more specific unfolded architectures are also available.

The Primal-Dual Network (PDNet) uses :class:`deepinv.models.PDNet_PrimalBlock` and
:class:`deepinv.models.PDNet_DualBlock` as building blocks for the primal and dual steps respectively.