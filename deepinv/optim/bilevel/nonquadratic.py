"""Smooth nonquadratic bilevel problem for estimator stress tests.

Lower level
    h(x, theta) = ||A2 x + A3 theta - b2||^2
                + (gamma / 2) ||x||^2
                + beta * sum_i log(cosh(x_i))

The log-cosh term is smooth and convex (Hessian ``sech^2(x_i) ∈ (0, 1]``),
so with ``gamma > 0`` the lower level is strongly convex with
``mu >= gamma``. The Hessian of ``h`` depends on ``x``, so the Newton
identity ``xbar - xhat = H(xbar)^{-1} grad_x h(xbar)`` is only approximate.
That is exactly the regime where the goal-oriented estimator may need a
larger safety factor than on the quadratic section 4.1 problem.

Upper level
    g(x) = ||A1 x - b1||^2
as in section 4.1 (quadratic, so ``G = 2 A1^T A1`` is exact).
"""

from __future__ import annotations

import torch

from .cg_utils import cg_solve


class NonQuadraticBilevel:
    """Smooth nonquadratic lower level with quadratic upper-level loss."""

    def __init__(
        self,
        A1: torch.Tensor,
        A2: torch.Tensor,
        A3: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        gamma: float = 1.0,
        beta: float = 0.5,
        gd_stepsize: float | None = None,
        gd_max_iter: int = 100_000,
    ):
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive for strong convexity, got {gamma}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.b1 = b1
        self.b2 = b2
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.n, self.d = A1.shape
        self.dtype = A1.dtype
        self.device = A1.device

        self._AtA = A2.T @ A2
        # Lower bound on strong convexity: data Hessian ≽ 0, log-cosh ≽ 0, gamma I.
        self.mu = self.gamma
        # Lipschitz of grad_x h: ||2 A2^T A2|| + gamma + beta (sech^2 <= 1).
        self.L_h = float(
            2.0 * torch.linalg.eigvalsh(self._AtA)[-1].item() + self.gamma + self.beta
        )
        AtA1 = A1.T @ A1
        self.L_g = float(2.0 * torch.linalg.eigvalsh(AtA1)[-1].item())
        self.G = 2.0 * AtA1  # Hessian of g
        self.J = 2.0 * (A2.T @ A3)
        self.J_norm = float(torch.linalg.matrix_norm(self.J, ord=2).item())
        self.L_H_inv = 0.0  # not used by DWR; certified path would need estimates
        self.L_J = 0.0

        if gd_stepsize is None:
            gd_stepsize = 1.0 / self.L_h
        self.gd_stepsize = float(gd_stepsize)
        self.gd_max_iter = int(gd_max_iter)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        r = self.A1 @ x - self.b1
        return torch.dot(r, r)

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (self.A1.T @ (self.A1 @ x - self.b1))

    def hess_g_matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.G @ v

    def residual_lower(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.A2 @ x + self.A3 @ theta - self.b2

    def grad_x_h(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        data = 2.0 * (self.A2.T @ self.residual_lower(x, theta))
        return data + self.gamma * x + self.beta * torch.tanh(x)

    def hess_x_matvec(
        self, x: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        data = 2.0 * (self._AtA @ v)
        # d/dx (beta tanh(x)) [v] = beta sech^2(x) * v
        sech2 = 1.0 / torch.cosh(x).pow(2)
        return data + self.gamma * v + self.beta * sech2 * v

    def mixed_jac_T_matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.J.T @ v

    def solve_lower(
        self,
        theta: torch.Tensor,
        eps: float,
        x_init: torch.Tensor | None = None,
        max_iter: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        if max_iter is None:
            max_iter = self.gd_max_iter
        if x_init is None:
            x = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        else:
            x = x_init.detach().clone()
        tol = eps * self.mu
        step = self.gd_stepsize
        grad = self.grad_x_h(x, theta)
        grad_norm = float(grad.norm().item())
        for _ in range(max_iter):
            if grad_norm <= tol:
                break
            x = x - step * grad
            grad = self.grad_x_h(x, theta)
            grad_norm = float(grad.norm().item())
        else:
            if grad_norm > tol:
                raise RuntimeError(
                    f"Nonquadratic GD failed: ||grad||={grad_norm} > tol={tol}"
                )
        return x, grad_norm

    def inexact_hypergradient(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        delta: float,
        max_cg_iter: int = 10_000,
        store_directions: bool = False,
    ):
        rhs = self.grad_g(x)

        def Hmv(v: torch.Tensor) -> torch.Tensor:
            return self.hess_x_matvec(x, v)

        cg = cg_solve(
            Hmv,
            rhs,
            tol=delta,
            max_iter=max_cg_iter,
            store_directions=store_directions,
        )
        z = -self.mixed_jac_T_matvec(cg.x)
        return z, cg

    def reference_hypergradient(
        self,
        theta: torch.Tensor,
        eps: float = 1e-12,
        delta: float = 1e-12,
        max_iter: int = 200_000,
    ) -> torch.Tensor:
        """High-accuracy reference hypergradient for error measurement."""
        x, _ = self.solve_lower(theta, eps=eps, max_iter=max_iter)
        z, _ = self.inexact_hypergradient(x, theta, delta=delta, max_cg_iter=100_000)
        return z

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        """Upper-level value at a high-accuracy lower-level solve (not closed form)."""
        x, _ = self.solve_lower(theta, eps=1e-12, max_iter=200_000)
        return self.g(x)

    def estimate_J_norm(self, x: torch.Tensor, theta: torch.Tensor, n_power: int = 1) -> float:
        return self.J_norm

    def update_lipschitz_estimates(self, x: torch.Tensor, theta: torch.Tensor) -> None:
        return None
