"""Instantiation B: convex saddle-point lower level (arXiv 2412.06436).

Lower level
    min_x max_y  h(x, y, K) = <K x, y> + g(x) - f*(y)

with g mu_g-strongly convex and f* mu_f*-strongly convex. Computable
distances (Lemma 2, equations 6a and 6b):

    ||x - xhat|| <= ||grad g(x) + K^T grad f(K x)|| / mu_g
    ||y - yhat|| <= ||grad f*(y) - K grad g*(-K^T y)|| / mu_f*

Adjoint distances follow the same strong-convexity fact (Lemma 3). The
hypergradient bound is Theorem 2 (displayed as equation 16 in the arXiv
HTML, constants 17a to 17d; the oracles brief numbers them 17 and 18a to
18d). Certified only when mu_g and mu_f* are known, not estimated.

Maps onto DeepInverse PDCP, and onto ADMM / DRS / HQS where a saddle form
is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .oracle import (
    HypergradientOracle,
    HypergradientState,
    LowerLevelState,
    strong_convexity_distance_bound,
)


def primal_distance_bound(
    grad_g_plus_KT_grad_f: float, mu_g: float
) -> float:
    r"""Equation 6a: :math:`\|x - \hat x\| \le \|\nabla g(x) + K^\top\nabla f(Kx)\| / \mu_g`."""
    return strong_convexity_distance_bound(grad_g_plus_KT_grad_f, mu_g)


def dual_distance_bound(
    grad_fstar_minus_K_grad_gstar: float, mu_fstar: float
) -> float:
    r"""Equation 6b: :math:`\|y - \hat y\| \le \|\nabla f^*(y) - K\nabla g^*(-K^\top y)\| / \mu_{f^*}`."""
    return strong_convexity_distance_bound(grad_fstar_minus_K_grad_gstar, mu_fstar)


def saddle_hypergradient_error_bound(
    eps_x: float,
    eps_y: float,
    delta_X: float,
    delta_Y: float,
    x_norm: float,
    y_norm: float,
    X_norm: float,
    Y_norm: float,
    C1X: float,
    C2X: float,
    C1Y: float,
    C2Y: float,
) -> float:
    r"""Theorem 2 hypergradient error bound (arXiv 2412.06436).

    For ``z = y ⊗ X + Y ⊗ x`` approximating ``grad L(K)``,

    .. math::

        \|z - \nabla\mathcal L(K)\|
        \le (C^Y_1 \|x\| + C^X_1 \|y\| + \|Y\|)\,\varepsilon^x
        + (C^Y_2 \|x\| + C^X_2 \|y\| + \|X\|)\,\varepsilon^y
        + \|y\|\,\delta^X + \|x\|\,\delta^Y
        + C^Y_1 (\varepsilon^x)^2 + C^X_2 (\varepsilon^y)^2
        + \delta^X \varepsilon^y + \delta^Y \varepsilon^x.

    The constants ``C`` are equations 17a to 17d of the arXiv HTML
    (18a to 18d in the oracles brief). All strong-convexity moduli that
    enter those constants must be known.
    """
    omega = (
        (C1Y * x_norm + C1X * y_norm + Y_norm) * eps_x
        + (C2Y * x_norm + C2X * y_norm + X_norm) * eps_y
        + y_norm * delta_X
        + x_norm * delta_Y
        + C1Y * (eps_x**2)
        + C2X * (eps_y**2)
        + delta_X * eps_y
        + delta_Y * eps_x
    )
    return float(omega)


def saddle_bound_constants(
    mu_g: float,
    mu_fstar: float,
    L_g: float,
    L_fstar: float,
    L_hess_gstar: float,
    L_hess_f: float,
    K_norm: float,
    X_norm: float,
    Y_norm: float,
    grad_ell1_norm: float,
    grad_ell2_norm: float,
    L1: float,
    L2: float,
) -> tuple[float, float, float, float]:
    r"""Constants (17a) to (17d) of Theorem 2, arXiv 2412.06436.

    .. math::

        C^X_1 = \frac{L_{\nabla^2 g^*}(L_{\nabla g})^3 \|X\| + L_1}{\mu_g}

        C^X_2 = \frac{L_{\nabla^2 f} L_{\nabla f^*}\|K\|
                (\|K\|\|X\| + \|\nabla\ell_2\|)}{\mu_g}
              + \frac{L_2 \|K\|}{\mu_g \mu_{f^*}}

        C^Y_1 = \frac{L_{\nabla^2 g^*} L_{\nabla g}\|K\|
                (\|K\|\|Y\| + \|\nabla\ell_1\|)}{\mu_{f^*}}
              + \frac{L_1 \|K\|}{\mu_g \mu_{f^*}}

        C^Y_2 = \frac{L_{\nabla^2 f}(L_{\nabla f^*})^3 \|Y\| + L_2}{\mu_{f^*}}
    """
    if mu_g <= 0.0 or mu_fstar <= 0.0:
        raise ValueError("mu_g and mu_fstar must be positive known constants")
    C1X = (L_hess_gstar * (L_g**3) * X_norm + L1) / mu_g
    C2X = (
        L_hess_f * L_fstar * K_norm * (K_norm * X_norm + grad_ell2_norm) / mu_g
        + L2 * K_norm / (mu_g * mu_fstar)
    )
    C1Y = (
        L_hess_gstar * L_g * K_norm * (K_norm * Y_norm + grad_ell1_norm) / mu_fstar
        + L1 * K_norm / (mu_g * mu_fstar)
    )
    C2Y = (L_hess_f * (L_fstar**3) * Y_norm + L2) / mu_fstar
    return float(C1X), float(C2X), float(C1Y), float(C2Y)


@dataclass
class QuadraticSaddleProblem:
    r"""Quadratic saddle bilevel problem with known closed form.

    Lower level (for fixed operator ``K``)
        min_x max_y  <K x, y> + (mu_g/2)||x||^2 - <p, x>
                     - (mu_fstar/2)||y||^2 + <q, y>

    which is equivalent to
        min_x  (1/(2 mu_fstar)) ||K x + q||^2 + (mu_g/2)||x||^2 - <p, x>
        (up to constants in y).

    Upper level
        min_K  (1/2) ||xhat(K) - x_target||^2

    with ``K`` stored as a flat vector of length ``n * d`` for MAID.
    Hessians of g and f* are constant, so ``L_hess_gstar = L_hess_f = 0``.
    """

    n: int
    d: int
    mu_g: float
    mu_fstar: float
    p: torch.Tensor
    q: torch.Tensor
    x_target: torch.Tensor
    dtype: torch.dtype = torch.float64
    device: torch.device | str = "cpu"
    pdhg_max_iter: int = 100_000

    def __post_init__(self) -> None:
        if self.mu_g <= 0.0 or self.mu_fstar <= 0.0:
            raise ValueError("mu_g and mu_fstar must be positive")
        self.device = torch.device(self.device)
        self.param_dim = self.n * self.d
        # g(x) = (mu_g/2)||x||^2 - <p,x> is mu_g-strongly convex and
        # mu_g-smooth. f*(y) = (mu_fstar/2)||y||^2 - <q,y> similarly.
        self.L_nabla_g = self.mu_g
        self.L_nabla_fstar = self.mu_fstar
        # Constant Hessians => Lipschitz of Hessians is zero.
        self.L_hess_gstar = 0.0
        self.L_hess_f = 0.0
        # Upper loss ell_1(x) = (1/2)||x - x_target||^2 is 1-smooth; ell_2 = 0.
        self.L1 = 1.0
        self.L2 = 0.0
        # Lip of grad_x ell_1 for U bounds equals 1.
        self._L_g_upper = 1.0

    def K_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return theta.reshape(self.n, self.d)

    def theta_from_K(self, K: torch.Tensor) -> torch.Tensor:
        return K.reshape(-1)

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu_g * x - self.p

    def grad_fstar(self, y: torch.Tensor) -> torch.Tensor:
        return self.mu_fstar * y - self.q

    def grad_gstar(self, v: torch.Tensor) -> torch.Tensor:
        # g*(v) = (1/(2 mu_g))||v + p||^2 + const, so grad g*(v) = (v + p)/mu_g.
        return (v + self.p) / self.mu_g

    def grad_f(self, u: torch.Tensor) -> torch.Tensor:
        # f(u) = (1/(2 mu_fstar))||u + q||^2 + const.
        return (u + self.q) / self.mu_fstar

    def primal_residual_norm(self, x: torch.Tensor, K: torch.Tensor) -> float:
        r"""``||grad g(x) + K^T grad f(K x)||`` used in equation 6a."""
        r = self.grad_g(x) + K.T @ self.grad_f(K @ x)
        return float(r.norm().item())

    def dual_residual_norm(self, y: torch.Tensor, K: torch.Tensor) -> float:
        r"""``||grad f*(y) - K grad g*(-K^T y)||`` used in equation 6b."""
        r = self.grad_fstar(y) - K @ self.grad_gstar(-K.T @ y)
        return float(r.norm().item())

    def closed_form_saddle(
        self, K: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact saddle point via the linear KKT system."""
        # From dual stationarity: mu_fstar y - q = K x  => y = (K x + q)/mu_fstar
        # From primal: mu_g x - p + K^T y = 0
        # => mu_g x + K^T (K x + q)/mu_fstar = p
        # => (mu_g I + K^T K / mu_fstar) x = p - K^T q / mu_fstar
        KtK = K.T @ K
        A = self.mu_g * torch.eye(self.d, dtype=self.dtype, device=self.device) + KtK / self.mu_fstar
        rhs = self.p - K.T @ self.q / self.mu_fstar
        x = torch.linalg.solve(A, rhs)
        y = (K @ x + self.q) / self.mu_fstar
        return x, y

    def closed_form_x(self, theta: torch.Tensor) -> torch.Tensor:
        return self.closed_form_saddle(self.K_from_theta(theta))[0]

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        x = self.closed_form_x(theta)
        return 0.5 * torch.dot(x - self.x_target, x - self.x_target)

    def exact_hypergradient(self, theta: torch.Tensor) -> torch.Tensor:
        """Exact hypergradient of (1/2)||xhat(K)-x_target||^2 w.r.t. flat K."""
        K = self.K_from_theta(theta).detach().clone().requires_grad_(True)
        x, _ = self.closed_form_saddle(K)
        loss = 0.5 * torch.dot(x - self.x_target, x - self.x_target)
        (grad_K,) = torch.autograd.grad(loss, K)
        return grad_K.reshape(-1).detach()

    def solve_pdhg(
        self,
        K: torch.Tensor,
        eps_x: float,
        eps_y: float,
        x_init: torch.Tensor | None = None,
        y_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """PDHG on the saddle until equations 6a and 6b meet the tolerances."""
        K_norm = float(torch.linalg.matrix_norm(K, ord=2).item())
        # Chambolle-Pock steps for strongly convex-concave saddle
        # (Chambolle & Pock 2016 style): tau sigma ||K||^2 < 1.
        # With strong convexity, use the nu schedule from the paper.
        nu = min(
            1.0,
            2.0 * (self.mu_g * self.mu_fstar) ** 0.5 / max(K_norm, 1e-12),
        )
        tau = nu / (2.0 * self.mu_g)
        sigma = nu / (2.0 * self.mu_fstar)
        theta_extrap = 1.0 / (1.0 + nu)

        x = (
            torch.zeros(self.d, dtype=self.dtype, device=self.device)
            if x_init is None
            else x_init.detach().clone()
        )
        y = (
            torch.zeros(self.n, dtype=self.dtype, device=self.device)
            if y_init is None
            else y_init.detach().clone()
        )
        x_bar = x.clone()

        for _ in range(self.pdhg_max_iter):
            # Dual step on f*: y <- prox_{sigma f*}(y + sigma K x_bar)
            # f*(y) = (mu_fstar/2)||y||^2 - <q,y>
            # prox_{sigma f*}(v) = (v + sigma q) / (1 + sigma mu_fstar)
            v = y + sigma * (K @ x_bar)
            y = (v + sigma * self.q) / (1.0 + sigma * self.mu_fstar)
            # Primal step on g: x <- prox_{tau g}(x - tau K^T y)
            # g(x) = (mu_g/2)||x||^2 - <p,x>
            # prox_{tau g}(v) = (v + tau p) / (1 + tau mu_g)
            w = x - tau * (K.T @ y)
            x_prev = x
            x = (w + tau * self.p) / (1.0 + tau * self.mu_g)
            x_bar = x + theta_extrap * (x - x_prev)

            rx = self.primal_residual_norm(x, K)
            ry = self.dual_residual_norm(y, K)
            dist_x = primal_distance_bound(rx, self.mu_g)
            dist_y = dual_distance_bound(ry, self.mu_fstar)
            if dist_x <= eps_x and dist_y <= eps_y:
                return x, y, dist_x, dist_y

        raise RuntimeError(
            f"PDHG failed to reach eps_x={eps_x}, eps_y={eps_y} "
            f"(got dist_x={dist_x}, dist_y={dist_y})."
        )

    def solve_adjoint_pdhg(
        self,
        K: torch.Tensor,
        x_tilde: torch.Tensor,
        y_tilde: torch.Tensor,
        delta_X: float,
        delta_Y: float,
        X_init: torch.Tensor | None = None,
        Y_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        r"""Solve the inexact adjoint saddle (ASI) by PDHG.

        For constant Hessians
            ASI: min_X max_Y <K X, Y> + (mu_g/2)||X||^2 - (mu_fstar/2)||Y||^2
                 + <grad ell_1(x_tilde), X>
        with ``grad ell_1 = x_tilde - x_target`` and ``ell_2 = 0``.
        """
        grad_ell1 = x_tilde - self.x_target
        K_norm = float(torch.linalg.matrix_norm(K, ord=2).item())
        nu = min(
            1.0,
            2.0 * (self.mu_g * self.mu_fstar) ** 0.5 / max(K_norm, 1e-12),
        )
        tau = nu / (2.0 * self.mu_g)
        sigma = nu / (2.0 * self.mu_fstar)
        theta_extrap = 1.0 / (1.0 + nu)

        X = (
            torch.zeros(self.d, dtype=self.dtype, device=self.device)
            if X_init is None
            else X_init.detach().clone()
        )
        Y = (
            torch.zeros(self.n, dtype=self.dtype, device=self.device)
            if Y_init is None
            else Y_init.detach().clone()
        )
        X_bar = X.clone()

        for _ in range(self.pdhg_max_iter):
            # Dual: quadratic (mu_fstar/2)||Y||^2, no linear term from ell_2.
            v = Y + sigma * (K @ X_bar)
            Y = v / (1.0 + sigma * self.mu_fstar)
            # Primal: (mu_g/2)||X||^2 + <grad_ell1, X>
            w = X - tau * (K.T @ Y) - tau * grad_ell1
            X_prev = X
            X = w / (1.0 + tau * self.mu_g)
            X_bar = X + theta_extrap * (X - X_prev)

            # Residual distances (Lemma 3) for constant Hessians:
            # B1 = mu_g I + K^T (1/mu_fstar) K
            # residual_X = ||B1 X + grad_ell1||  (ell_2 = 0)
            B1X = self.mu_g * X + K.T @ (K @ X) / self.mu_fstar
            res_X = float((B1X + grad_ell1).norm().item())
            dist_X = res_X / self.mu_g
            # B2 = K (1/mu_g) K^T + mu_fstar I
            # residual_Y = ||B2 Y + K (1/mu_g) grad_ell1||
            B2Y = K @ (K.T @ Y) / self.mu_g + self.mu_fstar * Y
            res_Y = float((B2Y + K @ (grad_ell1 / self.mu_g)).norm().item())
            dist_Y = res_Y / self.mu_fstar
            if dist_X <= delta_X and dist_Y <= delta_Y:
                return X, Y, dist_X, dist_Y

        raise RuntimeError(
            f"Adjoint PDHG failed to reach delta_X={delta_X}, delta_Y={delta_Y} "
            f"(got dist_X={dist_X}, dist_Y={dist_Y})."
        )

    def hypergradient_from_piggyback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        r"""``z = y ⊗ X + Y ⊗ x`` flattened to match ``theta``."""
        # dL/dK = y X^T + Y x^T
        grad_K = torch.outer(y, X) + torch.outer(Y, x)
        return grad_K.reshape(-1)


class SaddleHypergradientOracle(HypergradientOracle):
    """Saddle-point lower level with piggyback hypergradient and Theorem 2 bound."""

    def __init__(self, problem: QuadraticSaddleProblem):
        self.problem = problem
        self.n_lower_solves = 0
        self.n_hypergradients = 0

    @property
    def certified(self) -> bool:
        # mu_g and mu_fstar are constructor constants, not online estimates.
        return True

    @property
    def citation(self) -> str:
        return (
            "Bogensperger, Ehrhardt, Pock, Salehi, Wong, "
            "arXiv 2412.06436, Theorem 2 (Lemma 2 eqs 6a, 6b)"
        )

    @property
    def L_g(self) -> float:
        return self.problem._L_g_upper

    def reset_counters(self) -> None:
        self.n_lower_solves = 0
        self.n_hypergradients = 0

    def solve_lower_level(
        self,
        theta: torch.Tensor,
        eps: float,
        warm_start: LowerLevelState | None = None,
    ) -> LowerLevelState:
        K = self.problem.K_from_theta(theta)
        x_init = None if warm_start is None else warm_start.x
        y_init = None
        if warm_start is not None and "y" in warm_start.extras:
            y_init = warm_start.extras["y"]
        # Use the same eps for primal and dual distances.
        x, y, dist_x, dist_y = self.problem.solve_pdhg(
            K, eps_x=eps, eps_y=eps, x_init=x_init, y_init=y_init
        )
        self.n_lower_solves += 1
        # The certificate for U bounds is the primal distance.
        eps_cert = max(dist_x, dist_y)
        return LowerLevelState(
            x=x,
            eps=eps_cert,
            extras={
                "y": y,
                "dist_x": dist_x,
                "dist_y": dist_y,
                "K": K.detach(),
            },
        )

    def hypergradient(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        delta: float,
    ) -> HypergradientState:
        K = lower.extras["K"]
        y = lower.extras["y"]
        x = lower.x
        X, Y, dist_X, dist_Y = self.problem.solve_adjoint_pdhg(
            K, x, y, delta_X=delta, delta_Y=delta
        )
        self.n_hypergradients += 1
        z = self.problem.hypergradient_from_piggyback(x, y, X, Y)
        K_norm = float(torch.linalg.matrix_norm(K, ord=2).item())
        grad_ell1 = x - self.problem.x_target
        C1X, C2X, C1Y, C2Y = saddle_bound_constants(
            mu_g=self.problem.mu_g,
            mu_fstar=self.problem.mu_fstar,
            L_g=self.problem.L_nabla_g,
            L_fstar=self.problem.L_nabla_fstar,
            L_hess_gstar=self.problem.L_hess_gstar,
            L_hess_f=self.problem.L_hess_f,
            K_norm=K_norm,
            X_norm=float(X.norm().item()),
            Y_norm=float(Y.norm().item()),
            grad_ell1_norm=float(grad_ell1.norm().item()),
            grad_ell2_norm=0.0,
            L1=self.problem.L1,
            L2=self.problem.L2,
        )
        return HypergradientState(
            z=z,
            delta=delta,
            extras={
                "X": X,
                "Y": Y,
                "dist_X": dist_X,
                "dist_Y": dist_Y,
                "eps_x": float(lower.extras["dist_x"]),
                "eps_y": float(lower.extras["dist_y"]),
                "x_norm": float(x.norm().item()),
                "y_norm": float(y.norm().item()),
                "X_norm": float(X.norm().item()),
                "Y_norm": float(Y.norm().item()),
                "C1X": C1X,
                "C2X": C2X,
                "C1Y": C1Y,
                "C2Y": C2Y,
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
        # Use the realised residual distances, not just the requested eps/delta.
        return saddle_hypergradient_error_bound(
            eps_x=float(ex["eps_x"]),
            eps_y=float(ex["eps_y"]),
            delta_X=float(ex["dist_X"]),
            delta_Y=float(ex["dist_Y"]),
            x_norm=float(ex["x_norm"]),
            y_norm=float(ex["y_norm"]),
            X_norm=float(ex["X_norm"]),
            Y_norm=float(ex["Y_norm"]),
            C1X=float(ex["C1X"]),
            C2X=float(ex["C2X"]),
            C1Y=float(ex["C1Y"]),
            C2Y=float(ex["C2Y"]),
        )

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(x - self.problem.x_target, x - self.problem.x_target)

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.problem.x_target

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        return self.problem.f_closed_form(theta)
