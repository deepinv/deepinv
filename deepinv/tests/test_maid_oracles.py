"""Rung 2: HypergradientOracle seam, certification rule, saddle instantiation.

Smooth bound probe (144 configurations) must never under-estimate, matching
the supervisor's acceptance criterion for rung 1 after the refactor.
"""

from __future__ import annotations

import pytest
import torch

from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    QuadraticBilevelLS,
    QuadraticSaddleProblem,
    SaddleHypergradientOracle,
    SmoothHypergradientOracle,
    dual_distance_bound,
    hypergradient_error_bound,
    primal_distance_bound,
    saddle_hypergradient_error_bound,
    strong_convexity_distance_bound,
)
from deepinv.optim.bilevel.oracle import HypergradientOracle
from deepinv.tests.test_maid import _make_section41_problem


# ---------------------------------------------------------------------------
# Certification rule
# ---------------------------------------------------------------------------


class _UncertifiedOracle(HypergradientOracle):
    """Minimal non-certified stub used only to test the opt-in gate."""

    @property
    def certified(self) -> bool:
        return False

    @property
    def citation(self) -> str:
        return ""

    @property
    def L_g(self) -> float:
        return 1.0

    def solve_lower_level(self, theta, eps, warm_start=None):
        raise NotImplementedError

    def hypergradient(self, theta, lower, delta):
        raise NotImplementedError

    def error_bound(self, theta, lower, hyper, eps, delta):
        return 0.0

    def g(self, x):
        return torch.tensor(0.0)

    def grad_g(self, x):
        return torch.zeros_like(x)


def test_uncertified_oracle_rejected_by_default():
    with pytest.raises(ValueError, match="not certified"):
        MAID(_UncertifiedOracle())


def test_uncertified_oracle_accepted_with_opt_in():
    maid = MAID(_UncertifiedOracle(), allow_uncertified=True)
    assert maid.oracle.certified is False


def test_smooth_oracle_is_certified():
    problem, _, _ = _make_section41_problem()
    oracle = SmoothHypergradientOracle(problem)
    assert oracle.certified is True
    assert "2308.10098" in oracle.citation
    MAID(oracle)  # must not raise


def test_saddle_oracle_is_certified():
    problem = _make_saddle_problem()
    oracle = SaddleHypergradientOracle(problem)
    assert oracle.certified is True
    assert "2412.06436" in oracle.citation
    MAID(oracle)


# ---------------------------------------------------------------------------
# Strong-convexity distance fact
# ---------------------------------------------------------------------------


def test_strong_convexity_distance_bound_identity():
    """Lemma 1: ||x* - x|| <= ||grad Phi(x)|| / mu for a quadratic Phi."""
    mu = 2.5
    x_star = torch.tensor([1.0, -2.0], dtype=torch.float64)
    x = torch.tensor([0.3, 0.4], dtype=torch.float64)
    # Phi(x) = (mu/2)||x - x_star||^2, grad = mu (x - x_star)
    grad = mu * (x - x_star)
    bound = strong_convexity_distance_bound(float(grad.norm().item()), mu)
    true_dist = float((x - x_star).norm().item())
    assert bound >= true_dist - 1e-12
    assert abs(bound - true_dist) < 1e-12  # equality for quadratics


# ---------------------------------------------------------------------------
# Smooth 144-configuration probe (supervisor acceptance criterion)
# ---------------------------------------------------------------------------


def test_smooth_error_bound_never_underestimates_144():
    """omega never under-estimates true error across 144 configurations.

    Four seeds, three parameter scales, four eps, three delta. This is the
    probe the supervisor ran on rung 1; it must remain green after the
    oracle refactor.
    """
    seeds = [0, 1, 2, 3]
    scales = [0.5, 1.0, 2.0]
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4]
    delta_list = [1e-1, 1e-2, 1e-3]

    violations = 0
    ratios = []
    n_configs = 0

    for seed in seeds:
        for scale in scales:
            problem, theta0, _ = _make_section41_problem(seed=seed + 10)
            # Scale the starting point so the geometry varies.
            theta = scale * theta0
            z_exact = problem.exact_hypergradient(
                theta, problem.closed_form_x(theta)
            )
            for eps in eps_list:
                for delta in delta_list:
                    n_configs += 1
                    xbar, _ = problem.solve_lower(theta, eps=eps)
                    z, _, _ = problem.inexact_hypergradient(
                        xbar, theta, delta=delta
                    )
                    true_err = float((z - z_exact).norm().item())
                    omega = hypergradient_error_bound(
                        eps=eps,
                        delta=delta,
                        mu=problem.mu,
                        L_g=problem.L_g,
                        J_norm=problem.J_norm,
                        grad_g_norm=float(problem.grad_g(xbar).norm().item()),
                        L_H_inv=0.0,
                        L_J=0.0,
                    )
                    if omega < true_err - 1e-12:
                        violations += 1
                    if true_err > 1e-15:
                        ratios.append(omega / true_err)

    assert n_configs == 144
    assert violations == 0, f"{violations} of 144 bound violations"
    assert min(ratios) >= 1.0 - 1e-9
    # Loose is acceptable; under-estimate is not.
    assert max(ratios) > 1.0


def test_smooth_oracle_error_bound_matches_function():
    """Oracle.error_bound equals the free function on the same inputs."""
    problem, theta0, _ = _make_section41_problem()
    oracle = SmoothHypergradientOracle(problem)
    eps, delta = 1e-2, 1e-2
    lower = oracle.solve_lower_level(theta0, eps=eps)
    hyper = oracle.hypergradient(theta0, lower, delta=delta)
    omega_oracle = oracle.error_bound(theta0, lower, hyper, eps, delta)
    omega_fn = hypergradient_error_bound(
        eps=eps,
        delta=delta,
        mu=problem.mu,
        L_g=problem.L_g,
        J_norm=hyper.extras["J_norm"],
        grad_g_norm=hyper.extras["grad_g_norm"],
        L_H_inv=0.0,
        L_J=0.0,
    )
    assert abs(omega_oracle - omega_fn) < 1e-15


# ---------------------------------------------------------------------------
# Saddle-point certificates
# ---------------------------------------------------------------------------


def _make_saddle_problem(
    n: int = 6,
    d: int = 3,
    seed: int = 0,
    mu_g: float = 1.5,
    mu_fstar: float = 2.0,
) -> QuadraticSaddleProblem:
    gen = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    p = torch.randn(d, generator=gen, dtype=dtype)
    q = torch.randn(n, generator=gen, dtype=dtype)
    x_target = torch.randn(d, generator=gen, dtype=dtype)
    return QuadraticSaddleProblem(
        n=n,
        d=d,
        mu_g=mu_g,
        mu_fstar=mu_fstar,
        p=p,
        q=q,
        x_target=x_target,
        dtype=dtype,
    )


def test_saddle_primal_dual_distance_bounds():
    """Equations 6a and 6b dominate true distances at inexact iterates."""
    problem = _make_saddle_problem()
    K = torch.randn(problem.n, problem.d, dtype=torch.float64)
    xhat, yhat = problem.closed_form_saddle(K)

    # Perturb away from the saddle.
    x = xhat + 0.3 * torch.randn_like(xhat)
    y = yhat + 0.3 * torch.randn_like(yhat)

    rx = problem.primal_residual_norm(x, K)
    ry = problem.dual_residual_norm(y, K)
    bound_x = primal_distance_bound(rx, problem.mu_g)
    bound_y = dual_distance_bound(ry, problem.mu_fstar)
    true_x = float((x - xhat).norm().item())
    true_y = float((y - yhat).norm().item())
    assert bound_x >= true_x - 1e-12
    assert bound_y >= true_y - 1e-12


def test_saddle_pdhg_respects_requested_eps():
    problem = _make_saddle_problem()
    theta = torch.randn(problem.param_dim, dtype=torch.float64)
    K = problem.K_from_theta(theta)
    eps = 1e-3
    x, y, dist_x, dist_y = problem.solve_pdhg(K, eps_x=eps, eps_y=eps)
    xhat, yhat = problem.closed_form_saddle(K)
    assert float((x - xhat).norm().item()) <= eps + 1e-10
    assert float((y - yhat).norm().item()) <= eps + 1e-10
    assert dist_x <= eps + 1e-12
    assert dist_y <= eps + 1e-12


def test_saddle_error_bound_never_underestimates():
    """Theorem 2 omega dominates true piggyback hypergradient error."""
    problem = _make_saddle_problem()
    oracle = SaddleHypergradientOracle(problem)
    gen = torch.Generator().manual_seed(3)
    violations = 0
    n = 0
    for scale in [0.5, 1.0, 1.5]:
        for eps in [1e-2, 5e-3, 1e-3]:
            for delta in [1e-2, 1e-3]:
                theta = scale * torch.randn(
                    problem.param_dim, generator=gen, dtype=torch.float64
                )
                z_exact = problem.exact_hypergradient(theta)
                lower = oracle.solve_lower_level(theta, eps=eps)
                hyper = oracle.hypergradient(theta, lower, delta=delta)
                true_err = float((hyper.z - z_exact).norm().item())
                omega = oracle.error_bound(theta, lower, hyper, eps, delta)
                n += 1
                if omega < true_err - 1e-10:
                    violations += 1
    assert n == 18
    assert violations == 0, f"{violations} of {n} saddle bound violations"


def test_saddle_hypergradient_matches_autograd_when_exact():
    """Piggyback at the exact saddle and exact adjoint recovers grad L."""
    problem = _make_saddle_problem()
    theta = torch.randn(problem.param_dim, dtype=torch.float64)
    K = problem.K_from_theta(theta)
    x, y = problem.closed_form_saddle(K)
    # Exact adjoint at the exact saddle: solve ASI with tiny delta.
    X, Y, _, _ = problem.solve_adjoint_pdhg(
        K, x, y, delta_X=1e-10, delta_Y=1e-10
    )
    z = problem.hypergradient_from_piggyback(x, y, X, Y)
    z_exact = problem.exact_hypergradient(theta)
    assert torch.allclose(z, z_exact, atol=1e-5, rtol=1e-5)


def test_maid_with_saddle_oracle_reduces_loss():
    """MAID + saddle oracle decreases the closed-form upper level."""
    problem = _make_saddle_problem(n=4, d=2, seed=1)
    oracle = SaddleHypergradientOracle(problem)
    # Start from a modest random K.
    gen = torch.Generator().manual_seed(0)
    theta0 = 0.3 * torch.randn(problem.param_dim, generator=gen, dtype=torch.float64)
    f0 = float(problem.f_closed_form(theta0).item())
    config = MAIDConfig(
        eps0=1e-2,
        delta0=1e-2,
        alpha0=1e-2,
        rho=0.5,
        rho_bar=1.2,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=15,
        max_iter=15,
        tol=1e-5,
        g_convex=True,
    )
    maid = MAID(oracle, config)
    theta_final, history = maid.run(theta0)
    f_final = float(problem.f_closed_form(theta_final).item())
    assert f_final < f0
    # Inexact ||z|| is not monotone under changing eps/delta; the loss is.
    assert min(history["f_exact"]) < f0
    assert len(history["z_norm"]) >= 1


def test_saddle_bound_constants_zero_hess_lipschitz():
    """With constant Hessians the C constants reduce to the L1/L2 terms."""
    from deepinv.optim.bilevel.saddle import saddle_bound_constants

    C1X, C2X, C1Y, C2Y = saddle_bound_constants(
        mu_g=2.0,
        mu_fstar=3.0,
        L_g=2.0,
        L_fstar=3.0,
        L_hess_gstar=0.0,
        L_hess_f=0.0,
        K_norm=1.5,
        X_norm=0.4,
        Y_norm=0.5,
        grad_ell1_norm=0.2,
        grad_ell2_norm=0.0,
        L1=1.0,
        L2=0.0,
    )
    assert abs(C1X - 1.0 / 2.0) < 1e-15
    assert abs(C2Y - 0.0) < 1e-15
    assert C2X >= 0.0
    assert C1Y >= 0.0
