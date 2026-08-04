"""Section 4.1 linear least-squares bilevel problem.

    min_theta  f(theta) := ||A1 xhat(theta) - b1||^2
    s.t.       xhat(theta) := argmin_x ||A2 x + A3 theta - b2||^2

Matrices A_i are (n, d). The lower-level objective is strongly convex in x
whenever A2 has full column rank. H = grad^2_x h and J = grad^2_{x theta} h
are constant in x, so L_H_inv = L_J = 0.

This is the analytical benchmark of Salehi et al. (SIAM J. Math. Data Sci.
2025, section 4.1). Gradients use the un-halved squared residual of the
paper:

    grad_x h = 2 A2^T (A2 x + A3 theta - b2),
    H        = 2 A2^T A2,
    J        = 2 A2^T A3,
    grad g   = 2 A1^T (A1 x - b1),
    L_g      = 2 ||A1^T A1||_2.
"""

from __future__ import annotations

import torch

from .cg_utils import CGResult, cg_solve


class QuadraticBilevelLS:
    """Quadratic least-squares bilevel problem of section 4.1."""

    def __init__(
        self,
        A1: torch.Tensor,
        A2: torch.Tensor,
        A3: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        gd_stepsize: float | None = None,
        gd_max_iter: int = 100_000,
    ):
        if A1.ndim != 2 or A2.ndim != 2 or A3.ndim != 2:
            raise ValueError("A1, A2, A3 must be 2-D matrices")
        if A1.shape != A2.shape or A1.shape != A3.shape:
            raise ValueError("A1, A2, A3 must share the same shape (n, d)")
        if b1.shape != (A1.shape[0],) or b2.shape != (A2.shape[0],):
            raise ValueError("b1, b2 must have shape (n,)")

        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.b1 = b1
        self.b2 = b2
        self.n, self.d = A1.shape
        self.dtype = A1.dtype
        self.device = A1.device

        # Constant Hessian and mixed Jacobian.
        self._AtA = A2.T @ A2
        self.H = 2.0 * self._AtA
        self.J = 2.0 * (A2.T @ A3)

        evals = torch.linalg.eigvalsh(self.H)
        self.mu = float(evals[0].clamp_min(0.0).item())
        if self.mu <= 0.0:
            raise ValueError(
                "Lower-level Hessian is not positive definite: "
                "A2 must have full column rank."
            )
        self.L_h = float(evals[-1].item())

        # Smoothness of g(x) = ||A1 x - b1||^2: Lip(grad g) = 2 ||A1^T A1||_2.
        AtA1 = A1.T @ A1
        self.L_g = float(2.0 * torch.linalg.eigvalsh(AtA1)[-1].item())

        # Operator norm of J.
        # For a thin tall factor, ||J||_2 = 2 ||A2^T A3||_2.
        self.J_norm = float(torch.linalg.matrix_norm(self.J, ord=2).item())

        # Constant maps: Lipschitz estimates stay zero. Running maxima are
        # still exposed so MAID can treat them like the general case.
        self.L_H_inv = 0.0
        self.L_J = 0.0

        # GD step size: 1 / L for the lower-level smooth strongly convex problem.
        if gd_stepsize is None:
            gd_stepsize = 1.0 / self.L_h
        self.gd_stepsize = float(gd_stepsize)
        self.gd_max_iter = int(gd_max_iter)
        # Cumulative GD iterations across solve_lower calls (demo / profiling).
        self.n_gd_iters = 0

        # Closed-form helpers.
        self._P = torch.linalg.solve(self._AtA, A2.T)  # (d, n)

    # ------------------------------------------------------------------
    # Upper-level objective pieces
    # ------------------------------------------------------------------
    def g(self, x: torch.Tensor) -> torch.Tensor:
        r = self.A1 @ x - self.b1
        return torch.dot(r, r)

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (self.A1.T @ (self.A1 @ x - self.b1))

    def hess_g_matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Hessian of ``g`` applied to ``v``: ``G v = 2 A1^T A1 v``."""
        return 2.0 * (self.A1.T @ (self.A1 @ v))

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        return self.g(self.closed_form_x(theta))

    # ------------------------------------------------------------------
    # Lower-level objective pieces
    # ------------------------------------------------------------------
    def residual_lower(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.A2 @ x + self.A3 @ theta - self.b2

    def grad_x_h(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return 2.0 * (self.A2.T @ self.residual_lower(x, theta))

    def hess_x_matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.H @ v

    def mixed_jac_T_matvec(self, v: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`J^\top` to a vector in x-space: returns a theta-vector."""
        return self.J.T @ v

    # ------------------------------------------------------------------
    # Closed forms (for tests and verification only)
    # ------------------------------------------------------------------
    def closed_form_x(self, theta: torch.Tensor) -> torch.Tensor:
        return self._P @ (self.b2 - self.A3 @ theta)

    def closed_form_theta_star(self) -> torch.Tensor:
        r"""Exact upper-level minimiser of the quadratic :math:`f`."""
        M = self.A1 @ self._P @ self.A3  # (n, d)
        d_vec = self.A1 @ (self._P @ self.b2) - self.b1
        return torch.linalg.solve(M.T @ M, M.T @ d_vec)

    def exact_hypergradient(
        self, theta: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x is None:
            x = self.closed_form_x(theta)
        q = torch.linalg.solve(self.H, self.grad_g(x))
        return -self.mixed_jac_T_matvec(q)

    # ------------------------------------------------------------------
    # Numerical solvers used by MAID
    # ------------------------------------------------------------------
    def solve_lower(
        self,
        theta: torch.Tensor,
        eps: float,
        x_init: torch.Tensor | None = None,
        max_iter: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        r"""Gradient descent on :math:`h(\cdot, \theta)` until
        :math:`\|\nabla_x h\| \le \varepsilon \mu`.

        Warm-starts from ``x_init`` when provided. Returns ``(xbar, grad_norm)``.
        """
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
        n_it = 0
        for _ in range(max_iter):
            if grad_norm <= tol:
                break
            x = x - step * grad
            grad = self.grad_x_h(x, theta)
            grad_norm = float(grad.norm().item())
            n_it += 1
        else:
            if grad_norm > tol:
                raise RuntimeError(
                    f"Lower-level GD failed to reach ||grad|| <= {tol} "
                    f"(got {grad_norm}) in {max_iter} iterations."
                )
        self.n_gd_iters += n_it
        return x, grad_norm

    def inexact_hypergradient(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        delta: float,
        max_cg_iter: int = 10_000,
        store_directions: bool = False,
    ) -> tuple[torch.Tensor, CGResult]:
        r"""Solve :math:`H q = \nabla g(x)` to residual ``delta``, form
        :math:`z = -J^\top q`.

        Returns ``(z, cg_result)`` where ``cg_result.residual`` is
        ``grad g(x) - H q`` and optional Krylov directions are stored for
        recycling by the goal-oriented estimator.
        """
        rhs = self.grad_g(x)
        cg = cg_solve(
            self.hess_x_matvec,
            rhs,
            tol=delta,
            max_iter=max_cg_iter,
            store_directions=store_directions,
        )
        z = -self.mixed_jac_T_matvec(cg.x)
        return z, cg

    def estimate_J_norm(
        self, x: torch.Tensor, theta: torch.Tensor, n_power: int = 1
    ) -> float:
        r"""Return a valid upper bound on :math:`\|J\|`.

        The paper counts one power-method iteration toward the lower-level
        cost. A single power iteration systematically underestimates the
        spectral norm on this problem (observed gap of 5 to 40 percent
        across random starts), which would invalidate the a posteriori
        error bound. For the quadratic problem J is constant and small,
        so the exact spectral norm computed at construction is used as
        the bound. The power iteration is still performed when
        ``n_power >= 1`` so the matvec cost matches the paper's
        accounting; its value is discarded in favour of the exact norm.
        """
        if n_power >= 1:
            v = torch.randn(self.d, dtype=self.dtype, device=self.device)
            v = v / v.norm().clamp_min(1e-30)
            for _ in range(n_power):
                Jv = self.J @ v
                JTJv = self.J.T @ Jv
                nrm = JTJv.norm().clamp_min(1e-30)
                v = JTJv / nrm
            # Result deliberately unused: exact spectral norm is the bound.
            _ = self.J @ v
        return self.J_norm

    def update_lipschitz_estimates(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> None:
        """No-op for the quadratic problem (L_H_inv = L_J = 0)."""
        return None
