"""Algorithm 3.2 (INEXACT_GRADIENT) and the a posteriori hypergradient error bound.

The bound is Theorem 2.1 of Salehi, Mukherjee, Roberts and Ehrhardt,
"An adaptively inexact first-order method for bilevel optimisation with
application to hyperparameter learning", SIAM J. Math. Data Sci. 2025
(arXiv 2308.10098).

Let
    H(x, theta) = grad^2_x h(x, theta),
    J(x, theta) = grad^2_{x theta} h(x, theta),
    z = -J(xbar, theta)^T q
where q solves H(xbar, theta) q = grad g(xbar) to residual at most delta,
and xbar satisfies ||grad_x h(xbar, theta)|| <= eps * mu.

Then
    ||z - grad f(theta)|| <= omega,

with
    c(x) = L_g * ||J|| / mu
         + L_{H^{-1}} * ||grad g(x)|| * ||J||
         + L_J * ||grad g(x)|| / mu,

    omega = c(xbar) * eps
          + (||J|| / mu) * delta
          + (L_J * L_g / mu) * eps^2.

L_{H^{-1}} is the Lipschitz constant of H(x, theta)^{-1} in x (uniform in
theta). L_J is the Lipschitz constant of J in x. Both are zero when H and J
are independent of x, as in the section 4.1 quadratic problem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .quadratic_ls import QuadraticBilevelLS


def hypergradient_error_bound(
    eps: float,
    delta: float,
    mu: float,
    L_g: float,
    J_norm: float,
    grad_g_norm: float,
    L_H_inv: float = 0.0,
    L_J: float = 0.0,
) -> float:
    r"""Computable upper bound on :math:`\|z - \nabla f(\theta)\|`.

    See module docstring for the identity. All norms are Euclidean (vector)
    or spectral (operator).

    :param float eps: lower-level accuracy, with
        :math:`\|\tilde x - \hat x\| \le \varepsilon` implied by
        :math:`\|\nabla_x h\| \le \varepsilon \mu`.
    :param float delta: residual tolerance of the linear solve
        :math:`H q = \nabla g`, i.e. :math:`\|H q - \nabla g\| \le \delta`.
    :param float mu: strong-convexity modulus of :math:`h(\cdot, \theta)`.
    :param float L_g: Lipschitz constant of :math:`\nabla g`.
    :param float J_norm: operator norm :math:`\|J(\tilde x, \theta)\|`.
    :param float grad_g_norm: :math:`\|\nabla g(\tilde x)\|`.
    :param float L_H_inv: Lipschitz constant of :math:`H^{-1}` in :math:`x`.
    :param float L_J: Lipschitz constant of :math:`J` in :math:`x`.
    :return: scalar bound :math:`\omega`.
    """
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    c = (
        L_g * J_norm / mu
        + L_H_inv * grad_g_norm * J_norm
        + L_J * grad_g_norm / mu
    )
    omega = c * eps + (J_norm / mu) * delta + (L_J * L_g / mu) * (eps**2)
    return float(omega)


def inexact_gradient(
    problem: QuadraticBilevelLS,
    theta: torch.Tensor,
    eps: float,
    delta: float,
    eta: float,
    nu: float,
    x_init: torch.Tensor | None = None,
    max_refine: int = 40,
) -> tuple[torch.Tensor, float, float, float, torch.Tensor]:
    r"""Algorithm 3.2: adaptively refine an inexact hypergradient.

    Repeatedly solves the lower level to accuracy ``eps``, solves the
    Hessian system to residual ``delta``, forms
    :math:`z = -J^\top q`, and accepts when the error bound satisfies

    .. math::

        \omega \le (1 - \eta)\, \|z\|.

    Otherwise ``eps`` and ``delta`` are multiplied by ``nu`` in ``(0, 1)``
    and the process repeats.

    :param problem: bilevel problem exposing lower-level solve and
        hypergradient primitives.
    :param torch.Tensor theta: current upper-level parameter.
    :param float eps: initial lower-level accuracy.
    :param float delta: initial linear-solve residual tolerance.
    :param float eta: descent margin in ``(0, 1)``.
    :param float nu: accuracy reduction factor in ``(0, 1)``.
    :param torch.Tensor x_init: warm start for the lower-level solver.
    :param int max_refine: safety cap on the refinement loop.
    :return: ``(z, eps, delta, omega, xbar)``.
    """
    if not (0.0 < eta < 1.0):
        raise ValueError(f"eta must lie in (0, 1), got {eta}")
    if not (0.0 < nu < 1.0):
        raise ValueError(f"nu must lie in (0, 1), got {nu}")

    x = x_init
    z = torch.zeros_like(theta)
    omega = float("inf")

    for _ in range(max_refine):
        x, _ = problem.solve_lower(theta, eps=eps, x_init=x)
        z, _, _ = problem.inexact_hypergradient(x, theta, delta=delta)
        grad_g_norm = problem.grad_g(x).norm().item()
        # Running maxima of the Lipschitz estimates across upper-level use
        # are maintained by the problem (quadratic case: both stay zero).
        L_H_inv = problem.L_H_inv
        L_J = problem.L_J
        J_norm = problem.estimate_J_norm(x, theta)
        omega = hypergradient_error_bound(
            eps=eps,
            delta=delta,
            mu=problem.mu,
            L_g=problem.L_g,
            J_norm=J_norm,
            grad_g_norm=grad_g_norm,
            L_H_inv=L_H_inv,
            L_J=L_J,
        )
        z_norm = z.norm().item()
        if z_norm > 0.0 and omega <= (1.0 - eta) * z_norm:
            return z, eps, delta, omega, x
        eps = nu * eps
        delta = nu * delta

    raise RuntimeError(
        f"INEXACT_GRADIENT failed to meet the descent test after "
        f"{max_refine} refinements (omega={omega}, ||z||={z.norm().item()}, "
        f"eps={eps}, delta={delta})."
    )
