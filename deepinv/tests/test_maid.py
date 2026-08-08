"""Tests for MAID (Method of Adaptive Inexact Descent), rung 1.

Covers Algorithm 3.1 (MAID), Algorithm 3.2 (INEXACT_GRADIENT) with the
a posteriori error bound of Theorem 2.1, on the linear least-squares bilevel
problem of section 4.1 of Salehi et al., SIAM J. Math. Data Sci. 2025
(arXiv 2308.10098).
"""

from __future__ import annotations

import torch

from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    QuadraticBilevelLS,
    hypergradient_error_bound,
    inexact_gradient,
)


def _rand_well_conditioned(
    n: int,
    d: int,
    gen: torch.Generator,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Thin QR factor with singular values in [1, 2].

    Uncontrolled Gaussian draws make the upper-level Hessian badly
    conditioned (cond 1e4 to 1e5 for n~80, d~5), so a fixed-step or
    short MAID run cannot reach the known optimum. The problem form is
    identical to section 4.1; only the spectrum is controlled.
    """
    G = torch.randn(n, d, generator=gen, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(G, mode="reduced")
    s = torch.linspace(1.0, 2.0, d, dtype=dtype, device=device)
    return Q * s


def _make_section41_problem(
    n: int = 100,
    d: int = 4,
    seed: int = 1,
    dtype=torch.float64,
    device: str = "cpu",
) -> tuple[QuadraticBilevelLS, torch.Tensor, torch.Tensor]:
    """Build the section 4.1 least-squares bilevel problem and its closed-form optimum.

    Problem
    -------
    min_theta  f(theta) = ||A1 xhat(theta) - b1||^2
    s.t.       xhat(theta) = argmin_x ||A2 x + A3 theta - b2||^2

    Returns the problem, a starting theta_0 (all ones), and the closed-form
    minimiser theta_star of the quadratic upper level.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    A1 = _rand_well_conditioned(n, d, gen, dtype, device)
    A2 = _rand_well_conditioned(n, d, gen, dtype, device)
    A3 = _rand_well_conditioned(n, d, gen, dtype, device)
    x1 = torch.randn(d, generator=gen, dtype=dtype, device=device)
    x2 = torch.randn(d, generator=gen, dtype=dtype, device=device)
    theta_bar = torch.randn(d, generator=gen, dtype=dtype, device=device)
    y1 = torch.randn(n, generator=gen, dtype=dtype, device=device)
    y2 = torch.randn(n, generator=gen, dtype=dtype, device=device)
    b1 = A1 @ x1 + 0.01 * y1
    b2 = A2 @ x2 + A3 @ theta_bar + 0.01 * y2

    problem = QuadraticBilevelLS(A1=A1, A2=A2, A3=A3, b1=b1, b2=b2)
    theta0 = torch.ones(d, dtype=dtype, device=device)
    theta_star = problem.closed_form_theta_star()
    return problem, theta0, theta_star


def test_closed_form_optimum_is_stationary():
    """The analytical theta_star makes the exact hypergradient near zero."""
    problem, _, theta_star = _make_section41_problem()
    xhat = problem.closed_form_x(theta_star)
    z_exact = problem.exact_hypergradient(theta_star, xhat)
    assert z_exact.norm().item() < 1e-8


def test_exact_hypergradient_matches_autograd():
    """IFT hypergradient at a non-optimum theta matches torch autograd of f."""
    problem, theta0, _ = _make_section41_problem()
    theta = theta0.clone().requires_grad_(True)
    xhat = problem.closed_form_x(theta)
    f = problem.g(xhat)
    f.backward()
    z_exact = problem.exact_hypergradient(theta0, problem.closed_form_x(theta0))
    assert torch.allclose(z_exact, theta.grad, atol=1e-8, rtol=1e-7)


def test_error_bound_dominates_true_error():
    """omega from Theorem 2.1 is a valid upper bound on ||z - grad f||.

    For the quadratic problem H and J are constant, so L_H_inv = L_J = 0 and
    the bound reduces to
        omega = (L_g * ||J|| / mu) * eps + (||J|| / mu) * delta.
    """
    problem, theta0, _ = _make_section41_problem()
    # Deliberately inexact lower-level and CG solves.
    eps = 1e-2
    delta = 1e-2
    xbar, grad_norm = problem.solve_lower(theta0, eps=eps, x_init=None, max_iter=10_000)
    assert grad_norm <= eps * problem.mu + 1e-12

    z, cg = problem.inexact_hypergradient(xbar, theta0, delta=delta)
    residual = cg.residual_norm
    z_exact = problem.exact_hypergradient(theta0, problem.closed_form_x(theta0))
    true_err = (z - z_exact).norm().item()

    omega = hypergradient_error_bound(
        eps=eps,
        delta=delta,
        mu=problem.mu,
        L_g=problem.L_g,
        J_norm=problem.J_norm,
        grad_g_norm=problem.grad_g(xbar).norm().item(),
        L_H_inv=0.0,
        L_J=0.0,
    )
    assert (
        omega >= true_err - 1e-12
    ), f"error bound {omega} does not dominate true error {true_err}"
    assert omega > 0.0


def test_inexact_gradient_returns_descent_direction():
    """Certified Algorithm 3.2 path: omega <= (1 - eta) ||z||."""
    problem, theta0, _ = _make_section41_problem()
    eta = 0.5
    z, eps, delta, omega, xbar = inexact_gradient(
        problem,
        theta0,
        eps=1.0,
        delta=1.0,
        eta=eta,
        nu=0.5,
        x_init=None,
        check_descent_direction=True,
    )
    z_norm = z.norm().item()
    assert z_norm > 0.0
    assert omega <= (1.0 - eta) * z_norm + 1e-12
    z_exact = problem.exact_hypergradient(theta0, problem.closed_form_x(theta0))
    cos = torch.dot(z, z_exact) / (z.norm() * z_exact.norm())
    assert cos.item() > 0.0


def test_inexact_gradient_skip_descent_test_returns_nan_omega():
    """Default skip path never forms omega."""
    problem, theta0, _ = _make_section41_problem()
    z, eps, delta, omega, xbar = inexact_gradient(
        problem,
        theta0,
        eps=1e-2,
        delta=1e-2,
        eta=0.5,
        nu=0.5,
        check_descent_direction=False,
    )
    assert z.norm().item() > 0.0
    assert omega != omega  # nan


def _Lf(problem: QuadraticBilevelLS) -> float:
    """Spectral Lipschitz constant of the exact hypergradient for the quadratic f."""
    M = problem.A1 @ problem._P @ problem.A3
    return float(2.0 * torch.linalg.eigvalsh(M.T @ M)[-1].item())


def test_maid_recovers_known_optimum():
    """MAID converges to the closed-form upper-level minimiser on section 4.1."""
    problem, theta0, theta_star = _make_section41_problem()
    f0 = problem.f_closed_form(theta0).item()
    f_star = problem.f_closed_form(theta_star).item()
    assert f0 > f_star

    config = MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=1.0 / _Lf(problem),
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=30,
        max_iter=80,
        tol=1e-6,
        g_convex=True,
    )
    maid = MAID(problem, config)
    theta_final, history = maid.run(theta0)

    f_final = problem.f_closed_form(theta_final).item()
    assert f_final < f0
    rel_gap = (f_final - f_star) / max(abs(f_star), 1.0)
    assert (
        rel_gap < 1e-3
    ), f"relative gap to known optimum is {rel_gap}, f_final={f_final}, f_star={f_star}"
    param_err = (theta_final - theta_star).norm().item()
    assert param_err < 1e-2, f"||theta - theta_star|| = {param_err}"
    assert history["f_exact"][-1] <= history["f_exact"][0]
    assert min(history["f_exact"]) < 0.5 * f0


def test_maid_hypergradient_norm_decreases():
    """Stationarity measure ||z|| drops by at least an order of magnitude."""
    problem, theta0, _ = _make_section41_problem()
    config = MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=1.0 / _Lf(problem),
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=30,
        max_iter=60,
        tol=1e-6,
        g_convex=True,
    )
    maid = MAID(problem, config)
    _, history = maid.run(theta0)
    z0 = history["z_norm"][0]
    z_min = min(history["z_norm"])
    assert z_min < 0.1 * z0, f"z_norm from {z0} only down to {z_min}"


def test_u_bounds_sandwich_true_f():
    """U_lower <= f(theta) <= U_upper when xbar is within eps of xhat."""
    problem, theta0, _ = _make_section41_problem()
    eps = 1e-3
    xbar, _ = problem.solve_lower(theta0, eps=eps, max_iter=20_000)
    xhat = problem.closed_form_x(theta0)
    assert (xbar - xhat).norm().item() <= eps + 1e-10

    g_xbar = problem.g(xbar).item()
    grad_g_norm = problem.grad_g(xbar).norm().item()
    L_g = problem.L_g
    U_upper = g_xbar + grad_g_norm * eps + 0.5 * L_g * eps**2
    U_lower = g_xbar - grad_g_norm * eps - 0.5 * L_g * eps**2
    f_true = problem.f_closed_form(theta0).item()
    assert U_lower <= f_true + 1e-10
    assert f_true <= U_upper + 1e-10
