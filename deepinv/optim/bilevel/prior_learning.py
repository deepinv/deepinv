"""Bilevel learning of a DeepInverse prior weight.

Learns a scalar Tikhonov weight ``lambda = exp(theta)`` in

.. math::

    \\hat x(\\theta)
    = \\arg\\min_x
      \\tfrac12 \\|A x - y\\|^2
      + \\tfrac12 e^{\\theta}\\|x\\|^2

by minimising the supervised upper level

.. math::

    f(\\theta) = \\tfrac12 \\|\\hat x(\\theta) - x^\\star\\|^2.

The lower level is solved by a DeepInverse optimiser (``GD``, ``PGD`` or
``FISTA``) with residual-based stopping. The hypergradient uses the IFT
adjoint with CG on the Hessian

.. math::

    H = A^\\top A + e^{\\theta} I.

This is the first real-prior case: ``deepinv.optim.Tikhonov`` with a
learnable weight, driven by DeepInverse's own optimisers rather than a
hand-rolled loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import Tikhonov

from .base_optim_lower import (
    build_solver,
    residual_kind_for_solver,
    solve_base_optim,
)
from .cg_utils import CGResult, cg_solve
from .oracle import (
    HypergradientOracle,
    HypergradientState,
    LowerLevelState,
)

if TYPE_CHECKING:
    from deepinv.physics import Physics


@dataclass
class TikhonovWeightProblem:
    """Scalar Tikhonov-weight bilevel problem on a DeepInverse physics."""

    physics: Any  # deepinv.physics.Physics (avoid circular import at module load)
    y: torch.Tensor
    x_star: torch.Tensor
    solver: str = "GD"
    stepsize: float | None = None
    max_iter: int = 50_000
    # Optional Lipschitz of A^* A for stepsize selection.
    lipschitz_data: float | None = None

    def __post_init__(self) -> None:
        self.data_fidelity = L2(sigma=1.0)
        self.prior = Tikhonov()
        self.solver = self.solver.upper()
        self.residual_kind = residual_kind_for_solver(self.solver)
        self.n_gd_iters = 0
        self.n_lower_solves = 0
        self.n_hypergradients = 0
        # Upper level g(x) = 0.5 ||x - x_star||^2 is 1-smooth.
        self.L_g = 1.0
        self.L_H_inv = 0.0
        self.L_J = 0.0  # J depends on x linearly; exact DWR not claimed here.
        if self.lipschitz_data is None:
            self.lipschitz_data = self._estimate_data_lipschitz()
        # Lower bound on lambda_min(A^* A). For Denoising (A = Id) this is 1.
        self.mu_data = self._estimate_mu_data()
        if self.stepsize is None:
            # Will be refined per theta once lambda is known.
            self.stepsize = 1.0 / (self.lipschitz_data + 1.0)

    def _estimate_data_lipschitz(self, n_power: int = 20) -> float:
        """Power iteration for ||A^* A|| on the image domain."""
        x = torch.randn_like(self.x_star)
        x = x / x.flatten().norm().clamp_min(1e-30)
        for _ in range(n_power):
            Ax = self.physics.A(x)
            x = self.physics.A_adjoint(Ax)
            nrm = x.flatten().norm().clamp_min(1e-30)
            x = x / nrm
        Ax = self.physics.A(x)
        AtAx = self.physics.A_adjoint(Ax)
        return float(AtAx.flatten().norm().item())

    def _estimate_mu_data(self) -> float:
        """Lower bound on the strong-convexity contribution of the data term.

        For Denoising with ``A = Id`` this is 1. For general ``A`` we use 0
        (safe: residual_tol is tighter than necessary).
        """
        # Detect identity-like operators: one power step that preserves a random vector.
        x = torch.randn_like(self.x_star)
        Ax = self.physics.A(x)
        if Ax.shape == x.shape and torch.allclose(Ax, x, atol=1e-12, rtol=0.0):
            return 1.0
        return 0.0

    def lambda_from_theta(self, theta: torch.Tensor) -> float:
        return float(torch.exp(theta.reshape(())).item())

    def mu(self, theta: torch.Tensor) -> float:
        """Strong-convexity modulus: ``mu_data + exp(theta)``."""
        return self.mu_data + self.lambda_from_theta(theta)

    def _stepsize_for(self, lam: float) -> float:
        return 1.0 / (self.lipschitz_data + lam)

    def grad_x_h(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        lam = self.lambda_from_theta(theta)
        return self.data_fidelity.grad(x, self.y, self.physics) + lam * self.prior.grad(
            x
        )

    def hess_x_matvec(
        self, x: torch.Tensor, theta: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        lam = self.lambda_from_theta(theta)
        # H v = A^* A v + lambda v  (L2 data fidelity with sigma=1)
        return self.physics.A_adjoint(self.physics.A(v)) + lam * v

    def mixed_jac_T_matvec(
        self, x: torch.Tensor, theta: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        r"""Apply :math:`J^\top` where :math:`J = \partial^2_{x\theta} h`.

        For scalar ``theta`` with ``lambda = exp(theta)``,
        ``J dtheta = exp(theta) * x * dtheta``, so
        ``J^T v = exp(theta) * <x, v>``.
        """
        lam = self.lambda_from_theta(theta)
        return (lam * torch.sum(x * v)).reshape(theta.shape)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        diff = x - self.x_star
        return 0.5 * torch.sum(diff * diff)

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.x_star

    def hess_g_matvec(self, v: torch.Tensor) -> torch.Tensor:
        return v

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        """Upper-level value at a high-accuracy lower-level solve."""
        # Floor residual_tol against floating-point limits on the gradient.
        x, _ = self.solve_lower(theta, eps=1e-8, max_iter=50_000)
        return self.g(x)

    def solve_lower(
        self,
        theta: torch.Tensor,
        eps: float,
        x_init: torch.Tensor | None = None,
        max_iter: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        lam = self.lambda_from_theta(theta)
        mu = self.mu(theta)
        stepsize = self._stepsize_for(lam)
        model = build_solver(
            self.solver,
            data_fidelity=self.data_fidelity,
            prior=self.prior,
            lambda_reg=lam,
            stepsize=stepsize,
            max_iter=max_iter if max_iter is not None else self.max_iter,
        )
        # Floor the residual tolerance: when lambda is tiny, eps * mu can fall
        # below the floating-point residual floor of GD on ill-conditioned
        # physics (blur, weak regularisation), and the solver never terminates.
        residual_tol = max(float(eps) * float(mu), 1e-8)
        result = solve_base_optim(
            model,
            self.y,
            self.physics,
            residual_tol=residual_tol,
            residual_kind=self.residual_kind,
            x_init=x_init,
            max_iter=max_iter if max_iter is not None else self.max_iter,
        )
        self.n_lower_solves += 1
        self.n_gd_iters += result.n_iters
        if not result.converged:
            raise RuntimeError(
                f"{self.solver} failed to reach residual {residual_tol} "
                f"(got {result.residual}) in {result.n_iters} iterations."
            )
        return result.x, result.residual

    def inexact_hypergradient(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        delta: float,
        max_cg_iter: int = 10_000,
        store_directions: bool = False,
    ) -> tuple[torch.Tensor, CGResult]:
        rhs = self.grad_g(x)

        def Hmv(v: torch.Tensor) -> torch.Tensor:
            return self.hess_x_matvec(x, theta, v)

        cg = cg_solve(
            Hmv,
            rhs,
            tol=delta,
            max_iter=max_cg_iter,
            store_directions=store_directions,
        )
        z = -self.mixed_jac_T_matvec(x, theta, cg.x)
        self.n_hypergradients += 1
        return z, cg

    def estimate_J_norm(
        self, x: torch.Tensor, theta: torch.Tensor, n_power: int = 1
    ) -> float:
        # ||J|| = ||exp(theta) x|| for the map R -> image space.
        return float(
            (self.lambda_from_theta(theta) * x.flatten().norm()).item()
        )

    def update_lipschitz_estimates(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> None:
        return None

    def exact_hypergradient(self, theta: torch.Tensor) -> torch.Tensor:
        """High-accuracy reference hypergradient for tests."""
        x, _ = self.solve_lower(theta, eps=1e-12, max_iter=200_000)
        z, _ = self.inexact_hypergradient(x, theta, delta=1e-12, max_cg_iter=100_000)
        return z


class TikhonovWeightOracle(HypergradientOracle):
    """IFT hypergradient oracle for :class:`TikhonovWeightProblem`.

    Certified under Theorem 2.1 when ``mu = exp(theta)`` is treated as known
    at the current iterate (it is an explicit function of ``theta``, not an
    online estimate of strong convexity from residuals).
    """

    def __init__(self, problem: TikhonovWeightProblem):
        self.problem = problem
        self.n_lower_solves = 0
        self.n_hypergradients = 0

    @property
    def certified(self) -> bool:
        return True

    @property
    def citation(self) -> str:
        return (
            "Salehi et al., SIAM J. Math. Data Sci. 2025, Theorem 2.1; "
            f"lower level via deepinv.optim.{self.problem.solver} with "
            f"{self.problem.residual_kind} residual stopping"
        )

    @property
    def L_g(self) -> float:
        return self.problem.L_g

    def reset_counters(self) -> None:
        self.n_lower_solves = 0
        self.n_hypergradients = 0
        self.problem.n_lower_solves = 0
        self.problem.n_hypergradients = 0
        self.problem.n_gd_iters = 0

    def solve_lower_level(
        self,
        theta: torch.Tensor,
        eps: float,
        warm_start: LowerLevelState | None = None,
    ) -> LowerLevelState:
        x_init = None if warm_start is None else warm_start.x
        x, residual = self.problem.solve_lower(theta, eps=eps, x_init=x_init)
        self.n_lower_solves += 1
        return LowerLevelState(
            x=x,
            eps=eps,
            extras={
                "residual": residual,
                "mu": self.problem.mu(theta),
                "residual_kind": self.problem.residual_kind,
                "solver": self.problem.solver,
            },
        )

    def hypergradient(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        delta: float,
    ) -> HypergradientState:
        z, cg = self.problem.inexact_hypergradient(
            lower.x, theta, delta=delta, store_directions=True
        )
        self.n_hypergradients += 1
        return HypergradientState(
            z=z,
            delta=delta,
            extras={
                "q": cg.x,
                "residual_vec": cg.residual,
                "cg": cg,
                "x": lower.x,
                "theta": theta,
                "mu": self.problem.mu(theta),
                "L_H_inv": 0.0,
                "L_J": 0.0,
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
        from .smooth import smooth_hypergradient_error_bound

        x = hyper.extras.get("x", lower.x)
        th = hyper.extras.get("theta", theta)
        J_norm = self.problem.estimate_J_norm(x, th)
        grad_g_norm = float(self.grad_g(x).flatten().norm().item())
        return smooth_hypergradient_error_bound(
            eps=eps,
            delta=delta,
            mu=float(hyper.extras["mu"]),
            L_g=self.L_g,
            J_norm=J_norm,
            grad_g_norm=grad_g_norm,
            L_H_inv=0.0,
            L_J=0.0,
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
        return None
