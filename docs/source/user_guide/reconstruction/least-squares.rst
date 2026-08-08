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
    >>> physics = dinv.physics.Blur(filter=dinv.physics.functional.gaussian_blur(sigma=(2, 2)), noise_model=dinv.physics.GaussianNoise(sigma=0.01))
    >>> y = physics(x)
    >>> x_hat = physics.A_dagger(y)

The computation of the least-squares solution depends on the nature of the forward operator:

- If the forward operator is **non-linear**, the least-squares solution is computed via gradient descent using the :meth:`deepinv.physics.Physics.A_dagger` method of the physics operator.
- If the forward operator is **linear**, the :meth:`A_dagger <deepinv.physics.LinearPhysics.A_dagger>` method of the physics operator computes the least-squares solution efficiently.
  Internally, the library calls a linear least squares solver, such as :func:`Conjugate Gradient (CG) <deepinv.optim.linear.conjugate_gradient>`,
  :func:`Least Squares QR (LSQR) <deepinv.optim.linear.lsqr>`, :func:`Minimum Residual (MINRES) <deepinv.optim.linear.minres>`,
  or :func:`Biconjugate Gradient Stabilized (BiCGStab) <deepinv.optim.linear.bicgstab>`
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


Wiener Deconvolution
^^^^^^^^^^^^^^^^^^^^

When the forward operator is a circular convolution, :class:`deepinv.physics.BlurFFT` is diagonalised by the Fourier transform, and the damped least-squares problem above has the classical Wiener filter as its closed-form solution:

.. math::

    \hat{X}(f) = \frac{H^*(f)}{\lvert H(f) \rvert^2 + \lambda(f)} \, Y(f)

where :math:`H` is the transfer function of the blur, and :math:`\lambda` plays the role of a noise-to-signal power ratio :math:`S_n(f)/S_x(f)`: small where the signal dominates, large where the measurement is mostly noise.
Because the regularization is applied in the Fourier domain, :math:`\lambda` may vary with frequency, which the constant :math:`\ell_2` damping above cannot express.

This is available as a :class:`deepinv.models.Reconstructor`:

    >>> physics = dinv.physics.BlurFFT(img_size=x.shape[1:], filter=dinv.physics.functional.gaussian_blur(sigma=(2, 2)), noise_model=dinv.physics.GaussianNoise(sigma=0.01))
    >>> y = physics(x)
    >>> model = dinv.models.WienerDeconvolution(lambda_reg=0.01, prior="laplacian")
    >>> x_hat = model(y, physics)

or directly on the physics operator, in the same way as filtered back-projection in tomography:

    >>> x_hat = physics.A_dagger(y, wiener=True, lambda_reg=0.01)

The ``prior`` argument controls how :math:`\lambda` depends on frequency. With ``"flat"`` (or ``None``) it is constant, which recovers the damped least-squares solution above. With ``"laplacian"`` it is weighted by the power spectrum of a Laplacian filter, penalizing high frequencies more strongly. Passing a tensor for ``lambda_reg`` instead supplies the noise-to-signal ratio at every frequency directly.
See :class:`deepinv.models.WienerDeconvolution` for details.


Going Beyond Least Squares
^^^^^^^^^^^^^^^^^^^^^^^^^^

While these methods provide a first approach to solving inverse problems, they often fall short in terms of reconstruction quality, especially in challenging scenarios.
To achieve better results, we can incorporate prior knowledge about the signal or use data-driven approaches, such as learned regularizers or deep neural networks.
Check-out the :ref:`summary of reconstruction methods <reconstructors>` in the user guide.