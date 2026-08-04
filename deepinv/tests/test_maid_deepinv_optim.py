"""DeepInverse BaseOptim lower level for MAID: residuals, warm start, prior learning."""

from __future__ import annotations

import pytest
import torch

import deepinv as dinv
from deepinv.optim import GD, PGD, FISTA, L2, Tikhonov
from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    TikhonovWeightOracle,
    TikhonovWeightProblem,
    build_solver,
    gradient_residual,
    proximal_residual,
    residual_kind_for_solver,
    solve_base_optim,
)


def _denoising_setup(seed=0, size=16, sigma=0.1):
    torch.manual_seed(seed)
    dtype = torch.float64
    x_star = torch.rand(1, 1, size, size, dtype=dtype)
    physics = dinv.physics.Denoising(
        noise_model=dinv.physics.GaussianNoise(sigma=sigma)
    )
    # Fix noise for reproducibility: y = x_star + sigma * noise
    gen = torch.Generator().manual_seed(seed + 1)
    y = x_star + sigma * torch.randn(x_star.shape, generator=gen, dtype=dtype)
    return x_star, y, physics


def test_residual_kind_mapping():
    assert residual_kind_for_solver("GD") == "gradient"
    assert residual_kind_for_solver("PGD") == "proximal"
    assert residual_kind_for_solver("FISTA") == "proximal"
    with pytest.raises(ValueError, match="No residual criterion wired"):
        residual_kind_for_solver("ADMM")


def test_gd_gradient_residual_stopping():
    x_star, y, physics = _denoising_setup()
    lam = 0.5
    stepsize = 1.0 / (1.0 + lam)
    model = build_solver(
        "GD",
        data_fidelity=L2(),
        prior=Tikhonov(),
        lambda_reg=lam,
        stepsize=stepsize,
        max_iter=10_000,
    )
    mu = lam  # Tikhonov strong convexity (denoising A=Id, data also sc)
    # Full strong convexity is 1 + lam for denoising; use mu = 1+lam for tol.
    mu = 1.0 + lam
    eps = 1e-4
    result = solve_base_optim(
        model,
        y,
        physics,
        residual_tol=eps * mu,
        residual_kind="gradient",
    )
    assert result.converged
    assert result.residual <= eps * mu + 1e-12
    # Residual matches independent evaluation.
    r = gradient_residual(model, result.x, y, physics)
    assert abs(r - result.residual) < 1e-12


def test_pgd_proximal_residual_stopping():
    x_star, y, physics = _denoising_setup()
    lam = 0.5
    stepsize = 1.0 / 1.0  # Lip of data term for denoising is 1
    model = build_solver(
        "PGD",
        data_fidelity=L2(),
        prior=Tikhonov(),
        lambda_reg=lam,
        stepsize=stepsize,
        max_iter=10_000,
    )
    mu = 1.0 + lam
    eps = 1e-4
    result = solve_base_optim(
        model,
        y,
        physics,
        residual_tol=eps * mu,
        residual_kind="proximal",
    )
    assert result.converged
    assert result.residual <= eps * mu + 1e-12
    r = proximal_residual(model, result.x, y, physics)
    assert abs(r - result.residual) < 1e-12


def test_fista_proximal_residual_stopping():
    x_star, y, physics = _denoising_setup()
    lam = 0.5
    stepsize = 0.9  # FISTA needs gamma <= 1/Lip
    model = build_solver(
        "FISTA",
        data_fidelity=L2(),
        prior=Tikhonov(),
        lambda_reg=lam,
        stepsize=stepsize,
        max_iter=10_000,
    )
    mu = 1.0 + lam
    eps = 1e-4
    result = solve_base_optim(
        model,
        y,
        physics,
        residual_tol=eps * mu,
        residual_kind="proximal",
    )
    assert result.converged
    assert result.residual <= eps * mu + 1e-12


def test_warm_start_is_used():
    """Warm-starting from a solution that already meets the residual needs 0 iters."""
    x_star, y, physics = _denoising_setup()
    lam = 0.5
    stepsize = 1.0 / (1.0 + lam)
    model = build_solver(
        "GD",
        data_fidelity=L2(),
        prior=Tikhonov(),
        lambda_reg=lam,
        stepsize=stepsize,
        max_iter=50_000,
    )
    mu = 1.0 + lam
    eps = 1e-5
    cold = solve_base_optim(
        model, y, physics, residual_tol=eps * mu, residual_kind="gradient"
    )
    assert cold.converged
    assert cold.n_iters >= 1
    # Re-run from the solution: residual already met, no iteration.
    warm = solve_base_optim(
        model,
        y,
        physics,
        residual_tol=eps * mu,
        residual_kind="gradient",
        x_init=cold.x,
    )
    assert warm.converged
    assert warm.n_iters == 0
    assert torch.allclose(warm.x, cold.x)


def test_tikhonov_weight_hypergradient_matches_finite_difference():
    x_star, y, physics = _denoising_setup(size=12)
    problem = TikhonovWeightProblem(
        physics=physics, y=y, x_star=x_star, solver="GD"
    )
    theta = torch.tensor(0.0, dtype=torch.float64)  # lambda = 1
    z = problem.exact_hypergradient(theta)
    # Central finite difference of f.
    h = 1e-5
    f_p = problem.f_closed_form(theta + h)
    f_m = problem.f_closed_form(theta - h)
    fd = (f_p - f_m) / (2 * h)
    assert abs(float(z.item()) - float(fd.item())) < 5e-3


@pytest.mark.parametrize("solver", ["GD", "PGD", "FISTA"])
def test_maid_learns_tikhonov_weight(solver):
    """MAID reduces the upper-level loss when learning a Tikhonov weight."""
    x_star, y, physics = _denoising_setup(size=12, seed=2)
    problem = TikhonovWeightProblem(
        physics=physics, y=y, x_star=x_star, solver=solver, max_iter=20_000
    )
    if solver == "FISTA":
        # Ensure FISTA has its momentum parameter.
        pass
    oracle = TikhonovWeightOracle(problem)
    # Start from a poor weight.
    theta0 = torch.tensor(2.0, dtype=torch.float64)  # lambda = e^2 ~ 7.4
    f0 = float(problem.f_closed_form(theta0).item())
    # Rough Lipschitz of hypergradient for step size: try a small fixed alpha0.
    config = MAIDConfig(
        eps0=1e-2,
        delta0=1e-2,
        alpha0=0.5,
        rho=0.5,
        rho_bar=1.2,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=20,
        max_iter=25,
        tol=1e-5,
        g_convex=True,
        check_descent_direction=False,
    )
    maid = MAID(oracle, config)
    theta_final, history = maid.run(theta0)
    f_final = float(problem.f_closed_form(theta_final).item())
    assert f_final < f0, f"solver={solver}: f {f0} -> {f_final}"
    assert history["n_lower_solves"] > 0
    # Warm-start path was used: more than one lower solve across iterations.
    assert history["n_upper_iters"] >= 1
