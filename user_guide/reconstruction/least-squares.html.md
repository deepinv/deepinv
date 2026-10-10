<a id="least-squares"></a>

# Pseudoinverse

This section describes reconstruction methods that do not require priors or training, and can be used as baselines for more advanced reconstruction methods.

## Least Squares Reconstruction

A straightforward reconstruction method is to obtain a least-squares estimate of the signal $x$ from the measurements $y$ by solving:

$$
\hat x=\operatorname*{argmin}_x \, \lVert \forw{x}-y\rVert _2^2
$$

This solution can be computed using the [`A_dagger`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics.A_dagger) method of the physics operator associated with the forward model:

```pycon
>>> import deepinv as dinv
>>> from deepinv.utils import load_example
>>> x = load_example("butterfly.png")
>>> physics = dinv.physics.Blur(filter=dinv.physics.functional.gaussian_blur(sigma=(2, 2)), noise_model=dinv.physics.GaussianNoise(sigma=0.01))
>>> y = physics(x)
>>> x_hat = physics.A_dagger(y)
```

The computation of the least-squares solution depends on the nature of the forward operator:

- If the forward operator is **non-linear**, the least-squares solution is computed via gradient descent using the [`deepinv.physics.Physics.A_dagger()`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics.A_dagger) method of the physics operator.
- If the forward operator is **linear**, the [`A_dagger`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.A_dagger) method of the physics operator computes the least-squares solution efficiently.
  Internally, the library calls a linear least squares solver, such as [`Conjugate Gradient (CG)`](https://deepinv.org/api/stubs/deepinv.optim.linear.conjugate_gradient.html.md#deepinv.optim.linear.conjugate_gradient),
  [`Least Squares QR (LSQR)`](https://deepinv.org/api/stubs/deepinv.optim.linear.lsqr.html.md#deepinv.optim.linear.lsqr), [`Minimum Residual (MINRES)`](https://deepinv.org/api/stubs/deepinv.optim.linear.minres.html.md#deepinv.optim.linear.minres),
  or [`Biconjugate Gradient Stabilized (BiCGStab)`](https://deepinv.org/api/stubs/deepinv.optim.linear.bicgstab.html.md#deepinv.optim.linear.bicgstab)
  to compute the pseudo-inverse of the forward operator. See [`deepinv.optim.linear.least_squares()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares) for more details on the available solvers.
- If the forward operator is linear with a **closed-form singular value decomposition** (i.e., it inherits from [`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics)),
  the pseudo-inverse is computed directly in closed form for efficiency.

## Least Squares with $\ell_2$ Regularization

In some cases, the least-squares solution can be ill-posed or unstable, especially when the forward operator is ill-conditioned and the measurements are noisy.
To address this issue, an $\ell_2$ regularization term can be added to the least-squares objective, leading to a damped least-squares problem:

$$
\hat x=\operatorname*{argmin}_x \, \lVert \forw{x}-y\rVert _2^2 + \frac{1}{\gamma} \lVert x \rVert_2^2
$$

where $\gamma > 0$ is the damping parameter that controls the trade-off between data fidelity and regularization.

If the forward operator is linear, the damped least-squares solution can be computed efficiently using the [`deepinv.physics.LinearPhysics.prox_l2()`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.prox_l2) method of the physics operator.

```pycon
>>> x_hat = physics.prox_l2(z=0, y=y, gamma=.1)
```

As with the standard least-squares solution, if the forward operator has a closed-form singular value decomposition, the damped least-squares solution can be computed directly in closed form for efficiency.

## Going Beyond Least Squares

While these methods provide a first approach to solving inverse problems, they often fall short in terms of reconstruction quality, especially in challenging scenarios.
To achieve better results, we can incorporate prior knowledge about the signal or use data-driven approaches, such as learned regularizers or deep neural networks.
Check-out the [summary of reconstruction methods](https://deepinv.org/user_guide/reconstruction/introduction.html.md#reconstructors) in the user guide.
