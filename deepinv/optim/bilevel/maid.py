"""Algorithm 3.1: Method of Adaptive Inexact Descent (MAID).

Salehi, Mukherjee, Roberts and Ehrhardt, SIAM J. Math. Data Sci. 2025
(arXiv 2308.10098).

Hyperparameters
--------------
* ``rho`` in (0, 1), ``rho_bar`` > 1: reduce / increase the step size alpha.
* ``nu`` in (0, 1), ``nu_bar`` > 1: reduce / increase accuracies eps, delta.
* ``eta`` in (0, 1): descent margin for Algorithm 3.2.
* ``lambd`` with 0 < lambd < eta: Armijo fraction in the inexact line search.
* ``max_BT``: initial backtracking budget (grows when the line search fails).

Line search
-----------
With g being L_g-smooth,

    U_upper(x, eps) = g(x) + ||grad g(x)|| * eps + (L_g / 2) * eps^2
    U_lower(x, eps) = g(x) - ||grad g(x)|| * eps - (L_g / 2) * eps^2

    psi(alpha) = U_upper(xbar(theta+), eps+) - U_lower(xbar(theta), eps)
                 + lambd * alpha * ||z||^2

``psi(alpha) <= 0`` implies the exact sufficient decrease
``f(theta+) - f(theta) <= -lambd * alpha * ||z||^2``.

If ``g_convex`` is True, the tighter convex form of remark 3.6 is used:
``psi_tilde = psi - (L_g / 2) * eps^2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .hypergradient import inexact_gradient

if TYPE_CHECKING:
    from .quadratic_ls import QuadraticBilevelLS


@dataclass
class MAIDConfig:
    """Configuration for :class:`MAID`."""

    eps0: float = 1e-1
    delta0: float = 1e-1
    alpha0: float = 1e-2
    rho: float = 0.5
    rho_bar: float = 1.2
    nu: float = 0.5
    nu_bar: float = 1.1
    eta: float = 0.5
    lambd: float = 0.1
    max_BT: int = 20
    max_iter: int = 100
    tol: float = 1e-6
    g_convex: bool = False
    max_outer_BT: int = 50
    # Optional hard cap on how small eps / delta may become.
    eps_min: float = 1e-14
    delta_min: float = 1e-14


@dataclass
class MAID:
    """Method of Adaptive Inexact Descent (Algorithm 3.1)."""

    problem: QuadraticBilevelLS
    config: MAIDConfig = field(default_factory=MAIDConfig)

    def __post_init__(self) -> None:
        c = self.config
        if not (0.0 < c.rho < 1.0):
            raise ValueError(f"rho must lie in (0, 1), got {c.rho}")
        if not (c.rho_bar > 1.0):
            raise ValueError(f"rho_bar must be > 1, got {c.rho_bar}")
        if not (0.0 < c.nu < 1.0):
            raise ValueError(f"nu must lie in (0, 1), got {c.nu}")
        if not (c.nu_bar > 1.0):
            raise ValueError(f"nu_bar must be > 1, got {c.nu_bar}")
        if not (0.0 < c.lambd < c.eta < 1.0):
            raise ValueError(
                f"require 0 < lambd < eta < 1, got lambd={c.lambd}, eta={c.eta}"
            )
        if c.max_BT < 1:
            raise ValueError(f"max_BT must be >= 1, got {c.max_BT}")

    def run(
        self, theta0: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        """Run MAID from ``theta0``.

        Returns the final parameter and a history dict with keys
        ``f_exact``, ``z_norm``, ``omega``, ``eps``, ``delta``, ``alpha``.
        """
        c = self.config
        problem = self.problem
        theta = theta0.detach().clone()
        eps = float(c.eps0)
        delta = float(c.delta0)
        alpha = float(c.alpha0)

        history: dict[str, list[float]] = {
            "f_exact": [],
            "z_norm": [],
            "omega": [],
            "eps": [],
            "delta": [],
            "alpha": [],
        }

        x_warm: torch.Tensor | None = None

        for _k in range(c.max_iter):
            # Outer loop over growing backtracking budgets (Algorithm 3.1).
            accepted = False
            z = torch.zeros_like(theta)
            omega = float("inf")
            xbar = (
                x_warm
                if x_warm is not None
                else torch.zeros(problem.d, dtype=theta.dtype, device=theta.device)
            )
            alpha_k = alpha
            eps_k = eps
            delta_k = delta

            for j in range(c.max_BT, c.max_BT + c.max_outer_BT):
                z, eps_k, delta_k, omega, xbar = inexact_gradient(
                    problem,
                    theta,
                    eps=eps_k,
                    delta=delta_k,
                    eta=c.eta,
                    nu=c.nu,
                    x_init=x_warm,
                )
                problem.update_lipschitz_estimates(xbar, theta)

                z_norm_sq = float(torch.dot(z, z).item())
                z_norm = z_norm_sq**0.5
                if z_norm <= c.tol and omega <= c.tol:
                    history["f_exact"].append(float(problem.f_closed_form(theta).item()))
                    history["z_norm"].append(z_norm)
                    history["omega"].append(omega)
                    history["eps"].append(eps_k)
                    history["delta"].append(delta_k)
                    history["alpha"].append(alpha_k)
                    return theta, history

                # Prospective accuracy used in U_upper after a successful step.
                eps_next = c.nu_bar * eps_k
                g_old = float(problem.g(xbar).item())
                grad_g_old_norm = float(problem.grad_g(xbar).norm().item())
                U_lower = self._U_lower(g_old, grad_g_old_norm, eps_k)

                alpha_try = alpha_k
                line_ok = False
                x_trial = xbar
                for _i in range(j):
                    theta_trial = theta - alpha_try * z
                    # Solve lower level at the trial point (warm-started).
                    # Accuracy eps_k is at least as tight as eps_next when
                    # nu_bar > 1, so U_upper(., eps_next) remains valid.
                    x_trial, _ = problem.solve_lower(
                        theta_trial, eps=eps_k, x_init=xbar
                    )
                    g_new = float(problem.g(x_trial).item())
                    grad_g_new_norm = float(problem.grad_g(x_trial).norm().item())
                    U_upper = self._U_upper(g_new, grad_g_new_norm, eps_next)
                    psi = U_upper - U_lower + c.lambd * alpha_try * z_norm_sq
                    if c.g_convex:
                        psi = psi - 0.5 * problem.L_g * (eps_k**2)
                    if psi <= 0.0:
                        line_ok = True
                        alpha_k = alpha_try
                        break
                    alpha_try = c.rho * alpha_try

                if line_ok:
                    theta = theta - alpha_k * z
                    x_warm = x_trial
                    eps = max(c.nu_bar * eps_k, c.eps_min)
                    delta = max(c.nu_bar * delta_k, c.delta_min)
                    alpha = c.rho_bar * alpha_k
                    accepted = True
                    break

                # Backtracking failed at this budget: tighten accuracy.
                eps_k = max(c.nu * eps_k, c.eps_min)
                delta_k = max(c.nu * delta_k, c.delta_min)
                alpha_k = alpha  # reset step size for the next outer attempt

            if not accepted:
                raise RuntimeError(
                    "MAID line search failed: no step accepted within "
                    f"max_outer_BT={c.max_outer_BT} accuracy refinements."
                )

            history["f_exact"].append(float(problem.f_closed_form(theta).item()))
            history["z_norm"].append(float(z.norm().item()))
            history["omega"].append(omega)
            history["eps"].append(eps_k)
            history["delta"].append(delta_k)
            history["alpha"].append(alpha_k)

            if history["z_norm"][-1] <= c.tol:
                break

        return theta, history

    def _U_upper(self, g_val: float, grad_g_norm: float, eps: float) -> float:
        L = self.problem.L_g
        return g_val + grad_g_norm * eps + 0.5 * L * (eps**2)

    def _U_lower(self, g_val: float, grad_g_norm: float, eps: float) -> float:
        L = self.problem.L_g
        return g_val - grad_g_norm * eps - 0.5 * L * (eps**2)
