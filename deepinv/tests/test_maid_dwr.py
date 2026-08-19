"""Goal-oriented (DWR) hypergradient error estimator tests."""

from __future__ import annotations

import pytest
import torch

from deepinv.optim.bilevel import (
    GoalOrientedEstimator,
    GoalOrientedSmoothOracle,
    MAID,
    NonQuadraticBilevel,
    SmoothHypergradientOracle,
)
from deepinv.tests.test_maid import _make_section41_problem


def test_goal_oriented_requires_opt_in():
    problem, _, _ = _make_section41_problem()
    oracle = GoalOrientedSmoothOracle(problem)
    assert oracle.certified is False
    with pytest.raises(ValueError, match="not certified"):
        MAID(oracle)
    MAID(oracle, allow_uncertified=True)


def test_dwr_safety_factor_never_underestimates_quadratic():
    """On the section 4.1 problem, default DWR never under-estimates."""
    problem, theta0, _ = _make_section41_problem(n=80, d=5, seed=2)
    oracle = GoalOrientedSmoothOracle(problem, safety_factor=1.25, cg_budget=5)
    violations = 0
    n = 0
    ratios = []
    for scale in (0.5, 1.0, 2.0):
        theta = scale * theta0
        z_exact = problem.exact_hypergradient(theta)
        for eps in (1e-1, 1e-2, 1e-3):
            for delta in (1e-1, 1e-2, 1e-3):
                lower = oracle.solve_lower_level(theta, eps=eps)
                hyper = oracle.hypergradient(theta, lower, delta=delta)
                true_err = float((hyper.z - z_exact).norm().item())
                omega = oracle.error_bound(theta, lower, hyper, eps, delta)
                n += 1
                if true_err > 1e-15:
                    ratios.append(omega / true_err)
                    if omega < true_err * (1.0 - 1e-9):
                        violations += 1
    assert n == 27
    assert violations == 0, f"{violations}/{n} under-estimates"
    assert min(ratios) >= 1.0 - 1e-9
    assert sorted(ratios)[len(ratios) // 2] < 5.0


def test_raw_dwr_can_underestimate_but_safety_covers():
    problem, theta0, _ = _make_section41_problem(n=60, d=4, seed=0)
    raw = GoalOrientedSmoothOracle(problem, safety_factor=1.0, cg_budget=5)
    safe = GoalOrientedSmoothOracle(problem, safety_factor=1.25, cg_budget=5)
    theta = 1.0 * theta0
    z_exact = problem.exact_hypergradient(theta)
    lower = raw.solve_lower_level(theta, eps=1e-2)
    hyper = raw.hypergradient(theta, lower, delta=1e-2)
    true_err = float((hyper.z - z_exact).norm().item())
    omega_raw = raw.error_bound(theta, lower, hyper, 1e-2, 1e-2)
    omega_safe = safe.error_bound(theta, lower, hyper, 1e-2, 1e-2)
    assert abs(omega_safe - 1.25 * omega_raw) < 1e-12 * max(1.0, omega_safe)
    if true_err > 1e-15:
        assert omega_safe >= true_err * (1.0 - 1e-9)


def test_dwr_tighter_than_certified_on_quadratic():
    problem, theta0, _ = _make_section41_problem()
    certified = SmoothHypergradientOracle(problem)
    dwr = GoalOrientedSmoothOracle(problem, safety_factor=1.25, cg_budget=5)
    theta = theta0
    eps, delta = 1e-2, 1e-2
    lower = certified.solve_lower_level(theta, eps=eps)
    hyper_c = certified.hypergradient(theta, lower, delta=delta)
    hyper_d = dwr.hypergradient(theta, lower, delta=delta)
    omega_c = certified.error_bound(theta, lower, hyper_c, eps, delta)
    omega_d = dwr.error_bound(theta, lower, hyper_d, eps, delta)
    assert omega_d < omega_c


def test_estimator_rejects_safety_below_one():
    with pytest.raises(ValueError, match="safety_factor"):
        GoalOrientedEstimator(safety_factor=0.9)


def _make_nonquadratic_gaussian(n=80, d=20, seed=0, data_scale=1.0):
    """Nonquadratic problem with i.i.d. Gaussian factors, as in the nonlinearity sweep.

    ``data_scale`` multiplies ``b1`` and ``b2``. At scale 1, ``sech^2`` of the
    solution stays near 1 (barely nonlinear). At scales 5 and 20 the solution
    leaves the quadratic regime of log-cosh (supervisor table).
    """
    gen = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    mk = lambda: torch.randn(n, d, generator=gen, dtype=dtype)
    return NonQuadraticBilevel(
        A1=mk(),
        A2=mk(),
        A3=mk(),
        b1=data_scale * torch.randn(n, generator=gen, dtype=dtype),
        b2=data_scale * torch.randn(n, generator=gen, dtype=dtype),
        gamma=1.5,
        beta=0.5,
    )


def _sech2_spread(problem: NonQuadraticBilevel, theta: torch.Tensor) -> float:
    """Range of sech^2(x) over a high-accuracy lower-level solution."""
    x, _ = problem.solve_lower(theta, eps=1e-10, max_iter=200_000)
    sech2 = 1.0 / torch.cosh(x).pow(2)
    return float((sech2.max() - sech2.min()).item())


# Expected sech^2 spread bands from the supervisor nonlinearity sweep.
# Used as soft regime labels, not exact targets (finite d, n vary).
_SECH2_REGIME = {
    1.0: (0.0, 0.25),  # barely nonlinear
    5.0: (0.2, 1.01),  # genuinely nonlinear
    20.0: (0.5, 1.01),  # strongly nonlinear
}


@pytest.mark.parametrize("data_scale", [1.0, 5.0, 20.0])
def test_nonquadratic_dwr_across_nonlinearity_scales(data_scale):
    """DWR with safety 1.25 does not under-estimate at three nonlinearity levels.

    Records the ``sech^2`` spread so a reader can see which regime is covered.
    A nonlinearity test that silently runs in the linear regime is not coverage.
    """
    problem = _make_nonquadratic_gaussian(n=80, d=20, seed=0, data_scale=data_scale)
    theta0 = torch.zeros(problem.d, dtype=torch.float64)
    sech2_spread = _sech2_spread(problem, theta0)
    lo, hi = _SECH2_REGIME[data_scale]
    assert lo <= sech2_spread <= hi, (
        f"data_scale={data_scale}: sech2_spread={sech2_spread:.4f} "
        f"outside expected regime band [{lo}, {hi}]"
    )

    oracle = GoalOrientedSmoothOracle(problem, safety_factor=1.25, cg_budget=5)
    under = 0
    n = 0
    ratios = []
    for tscale in (0.1, 1.0):
        gen = torch.Generator().manual_seed(0)
        theta = tscale * torch.randn(problem.d, generator=gen, dtype=torch.float64)
        z_exact = problem.reference_hypergradient(theta, eps=1e-10, delta=1e-10)
        for eps in (1e-2, 1e-3):
            for delta in (1e-2, 1e-3):
                lower = oracle.solve_lower_level(theta, eps=eps)
                hyper = oracle.hypergradient(theta, lower, delta=delta)
                true_err = float((hyper.z - z_exact).norm().item())
                omega = oracle.error_bound(theta, lower, hyper, eps, delta)
                n += 1
                if true_err > 1e-14:
                    ratios.append(omega / true_err)
                    if omega < true_err * (1.0 - 1e-9):
                        under += 1

    assert n == 8
    assert under == 0, (
        f"data_scale={data_scale} sech2_spread={sech2_spread:.4f}: "
        f"safety 1.25 under-estimated {under}/{n}, "
        f"min ratio={min(ratios) if ratios else float('nan'):.4f}, "
        f"median={sorted(ratios)[len(ratios)//2] if ratios else float('nan'):.4f}"
    )
    assert min(ratios) >= 1.0 - 1e-9
