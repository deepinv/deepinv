.. _least_squares:

Pseudoinverse
=============

This section describes reconstruction methods that do not require priors or training, and can be used as baselines for more advanced reconstruction methods.


Least Squares Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A straightforward reconstruction method is to obtain a least-squares estimate of the signal :math:`x` from the measurements :math:`y` by solving:

.. math::

    \hat x=\operatorname*{argmin}_x \, \lVert \forw{x}-y\rVert _2^2


This solution can be computed using the :meth:`A_dagger <deepinv.physics.Physics.A_dagger>` method of the physics operator associated with the forward model:

    >>> import deepinv as dinv
    >>> from deepinv.utils import load_example
    >>> x = load_example("butterfly.png")
    >>> physics = dinv.physics.Blur(filter=dinv.physics.blur.gaussian_blur(2), noise_model=dinv.physics.GaussianNoise(sigma=0.01))
    >>> y = physics(x)
    >>> x_hat = physics.A_dagger(y)

The computation of the least-squares solution depends on the nature of the forward operator:
- If the forward operator is **non-linear**, the least-squares solution is computed via gradient descent using the :meth:`deepinv.physics.Physics.A_dagger` method of the physics operator.
- If the forward operator is linear, the :meth:deepinv.physics.LinearPhysics.A_dagger method of the physics operator computes the least-squares solution efficiently.
  Internally, the library calls a linear least squares solver, such as :func:`Conjugate Gradient (CG) <deepinv.optim.linear.conjugate_gradient>`,
  :func:`Least Squares QR (LSQR) <deepinv.optim.linear.lsqr>`, :func:`Minimum Residual (MINRES) <deepinv.optim.linear.minres>`, or :func:`Biconjugate Gradient Stabilized (BiCGStab) <deepinv.optim.linear.bicgstab>`.
  to compute the pseudo-inverse of the forward operator. See :func:`deepinv.optim.linear.least_squares` for more details on the available solvers.
- If the forward operator is linear with a **closed-form singular value decomposition** (i.e., it inherits from :class:`deepinv.physics.DecomposablePhysics`),
  the pseudo-inverse is computed directly in closed form for efficiency.


Least Squares with :math:`\ell_2` Regularization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some cases, the least-squares solution can be ill-posed or unstable, especially when the forward operator is ill-conditioned and the measurements are noisy.
To address this issue, an :math:`\ell_2` regularization term can be added to the least-squares objective, leading to a damped least-squares problem:

.. math::

    \hat x=\operatorname*{argmin}_x \, \lVert \forw{x}-y\rVert _2^2 + \frac{1}{\gamma} \lVert x \rVert_2^2

where :math:`\gamma > 0` is the damping parameter that controls the trade-off between data fidelity and regularization.

If the forward operator is linear, the damped least-squares solution can be computed efficiently using the :meth:`deepinv.physics.LinearPhysics.prox_l2` method of the physics operator.

    >>> x_hat = physics.prox_l2(z=0, y=y, gamma=.1)

As with the standard least-squares solution, if the forward operator has a closed-form singular value decomposition, the damped least-squares solution can be computed directly in closed form for efficiency.


Going Beyond Least Squares
^^^^^^^^^^^^^^^^^^^^^^^^^^

While these methods provide a first approach to solving inverse problems, they often fall short in terms of reconstruction quality, especially in challenging scenarios.
To achieve better results, we can incorporate prior knowledge about the signal or use data-driven approaches, such as learned regularizers or deep neural networks.
Check-out the :ref:`summary of reconstruction methods <reconstructors>` in the user guide.