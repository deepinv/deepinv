"""Instantiation A: smooth lower level (MAID Theorem 2.1 / arXiv 2308.10098).

Stopping rule
    ||grad_x h(xbar, theta)|| <= eps * mu
implies
    ||xbar - xhat|| <= eps
by the strong-convexity distance fact. The hypergradient is the IFT adjoint
solved by CG to residual delta, and the error bound is Theorem 2.1.

Maps onto DeepInverse GD, and onto PGD / FISTA when the prior is smooth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .oracle import HypergradientOracle, HypergradientState, LowerLevelState

if TYPE_CHECKING:
    from .quadratic_ls import QuadraticBilevelLS


def smooth_hypergradient_error_bound(
    eps: float,
    delta: float,
    mu: float,
    L_g: float,
    J_norm: float,
    grad_g_norm: float,
    L_H_inv: float = 0.0,
    L_J: float = 0.0,
) -> float:
    r"""Theorem 2.1 of Salehi et al., SIAM J. Math. Data Sci. 2025.

    .. math::

        c(x) = \frac{L_g \|J\|}{\mu}
             + L_{H^{-1}} \|\nabla g(x)\| \|J\|
             + \frac{L_J \|\nabla g(x)\|}{\mu},

        \omega = c(\tilde x)\,\varepsilon
              + \frac{\|J\|}{\mu}\,\delta
              + \frac{L_J L_g}{\mu}\,\varepsilon^2.

    ``mu`` must be a known strong-convexity modulus. An estimated ``mu``
    makes this bound non-certified even though the identity is a theorem.
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


# Backward-compatible alias used by existing tests and the 144-config probe.
hypergradient_error_bound = smooth_hypergradient_error_bound


class SmoothHypergradientOracle(HypergradientOracle):
    """Smooth lower level with IFT+CG hypergradient and Theorem 2.1 bound."""

    def __init__(self, problem: QuadraticBilevelLS):
        self.problem = problem

    @property
    def certified(self) -> bool:
        # mu is the exact smallest Hessian eigenvalue of the quadratic
        # lower level, not an online estimate.
        return True

    @property
    def citation(self) -> str:
        return (
            "Salehi, Mukherjee, Roberts, Ehrhardt, "
            "SIAM J. Math. Data Sci. 2025, Theorem 2.1 (arXiv 2308.10098)"
        )

    @property
    def L_g(self) -> float:
        return self.problem.L_g

    def solve_lower_level(
        self,
        theta: torch.Tensor,
        eps: float,
        warm_start: LowerLevelState | None = None,
    ) -> LowerLevelState:
        x_init = None if warm_start is None else warm_start.x
        x, grad_norm = self.problem.solve_lower(theta, eps=eps, x_init=x_init)
        return LowerLevelState(
            x=x,
            eps=eps,
            extras={"grad_norm": grad_norm, "mu": self.problem.mu},
        )

    def hypergradient(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        delta: float,
    ) -> HypergradientState:
        z, q, residual = self.problem.inexact_hypergradient(
            lower.x, theta, delta=delta
        )
        J_norm = self.problem.estimate_J_norm(lower.x, theta)
        grad_g_norm = float(self.problem.grad_g(lower.x).norm().item())
        return HypergradientState(
            z=z,
            delta=delta,
            extras={
                "q": q,
                "residual": residual,
                "J_norm": J_norm,
                "grad_g_norm": grad_g_norm,
                "L_H_inv": self.problem.L_H_inv,
                "L_J": self.problem.L_J,
                "mu": self.problem.mu,
            },
        )

    def error_bound(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        hyper: HypergradientState,
        eps: float,
        delta: float,
    ) -> float:
        ex = hyper.extras
        return smooth_hypergradient_error_bound(
            eps=eps,
            delta=delta,
            mu=float(ex["mu"]),
            L_g=self.problem.L_g,
            J_norm=float(ex["J_norm"]),
            grad_g_norm=float(ex["grad_g_norm"]),
            L_H_inv=float(ex["L_H_inv"]),
            L_J=float(ex["L_J"]),
        )

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return self.problem.g(x)

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        return self.problem.grad_g(x)

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        return self.problem.f_closed_form(theta)

    def update_lipschitz_estimates(
        self, lower: LowerLevelState, theta: torch.Tensor
    ) -> None:
        self.problem.update_lipschitz_estimates(lower.x, theta)


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
    """Algorithm 3.2 on a smooth quadratic problem (backward-compatible API)."""
    oracle = SmoothHypergradientOracle(problem)
    warm = None if x_init is None else LowerLevelState(x=x_init, eps=eps)
    return inexact_gradient_from_oracle(
        oracle, theta, eps, delta, eta, nu, warm_start=warm, max_refine=max_refine
    )


def inexact_gradient_from_oracle(
    oracle: HypergradientOracle,
    theta: torch.Tensor,
    eps: float,
    delta: float,
    eta: float,
    nu: float,
    warm_start: LowerLevelState | None = None,
    max_refine: int = 40,
) -> tuple[torch.Tensor, float, float, float, LowerLevelState]:
    r"""Algorithm 3.2 against an arbitrary :class:`HypergradientOracle`.

    Accepts when ``omega <= (1 - eta) * ||z||``.
    """
    if not (0.0 < eta < 1.0):
        raise ValueError(f"eta must lie in (0, 1), got {eta}")
    if not (0.0 < nu < 1.0):
        raise ValueError(f"nu must lie in (0, 1), got {nu}")

    lower = warm_start
    z = torch.zeros_like(theta)
    omega = float("inf")

    for _ in range(max_refine):
        lower = oracle.solve_lower_level(theta, eps=eps, warm_start=lower)
        hyper = oracle.hypergradient(theta, lower, delta=delta)
        z = hyper.z
        omega = oracle.error_bound(theta, lower, hyper, eps=eps, delta=delta)
        z_norm = float(z.norm().item())
        if z_norm > 0.0 and omega <= (1.0 - eta) * z_norm:
            return z, eps, delta, omega, lower
        eps = nu * eps
        delta = nu * delta

    raise RuntimeError(
        f"INEXACT_GRADIENT failed to meet the descent test after "
        f"{max_refine} refinements (omega={omega}, ||z||={z.norm().item()}, "
        f"eps={eps}, delta={delta}, certified={oracle.certified})."
    )
