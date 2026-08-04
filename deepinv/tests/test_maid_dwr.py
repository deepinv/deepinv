"""Goal-oriented (DWR) hypergradient error estimator tests."""

from __future__ import annotations

import pytest
import torch

from deepinv.optim.bilevel import (
    GoalOrientedEstimator,
    GoalOrientedSmoothOracle,
    MAID,
    MAIDConfig,
    NonQuadraticBilevel,
    QuadraticBilevelLS,
    SmoothHypergradientOracle,
    hypergradient_error_bound,
)
from deepinv.tests.test_maid import _make_section41_problem, _rand_well_conditioned


def test_goal_oriented_requires_opt_in():
    problem, _, _ = _make_section41_problem()
    oracle = GoalOrientedSmoothOracle(problem)
    assert oracle.certified is False
    with pytest.raises(ValueError, match="not certified"):
        MAID(oracle)
    MAID(oracle, allow_uncertified=True)


def test_dwr_safety_factor_never_underestimates_quadratic():
    """On the section 4.1 problem, default DWR never under-estimates.

    Smaller sweep than the supervisor's dim-200 run, but same estimator.
    """
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
    # Should be much tighter than the certified median ~24 on large instances.
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
    # Safe bound must dominate when true error is positive.
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


def _make_nonquadratic(n=50, d=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    A1 = _rand_well_conditioned(n, d, gen, dtype, "cpu")
    A2 = _rand_well_conditioned(n, d, gen, dtype, "cpu")
    A3 = _rand_well_conditioned(n, d, gen, dtype, "cpu")
    b1 = torch.randn(n, generator=gen, dtype=dtype)
    b2 = torch.randn(n, generator=gen, dtype=dtype)
    return NonQuadraticBilevel(
        A1=A1, A2=A2, A3=A3, b1=b1, b2=b2, gamma=1.5, beta=0.5
    )


def test_nonquadratic_dwr_underestimation_rate_reported():
    """Measure DWR on nonquadratic h; do not claim the quadratic safety factor.

    Records under-estimation rates for safety factors 1.0 and 1.25 so the
    docstring recommendation can be decided from data rather than hope.
    """
    problem = _make_nonquadratic()
    theta0 = torch.ones(problem.d, dtype=torch.float64)
    results = {}
    for sf in (1.0, 1.25, 2.0, 5.0):
        oracle = GoalOrientedSmoothOracle(
            problem, safety_factor=sf, cg_budget=5
        )
        under = 0
        n = 0
        ratios = []
        for scale in (0.5, 1.0):
            theta = scale * theta0
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
        results[sf] = {
            "under": under,
            "n": n,
            "median": sorted(ratios)[len(ratios) // 2] if ratios else float("nan"),
            "min": min(ratios) if ratios else float("nan"),
            "max": max(ratios) if ratios else float("nan"),
        }

    # Always expose the measurement (this is the decision data).
    print("\nnonquadratic DWR sweep:")
    for sf, r in results.items():
        print(
            f"  safety={sf}: under={r['under']}/{r['n']} "
            f"median={r['median']:.3f} min={r['min']:.3f} max={r['max']:.3f}"
        )

    assert results[1.0]["n"] == 8
    # Raw DWR under-estimates on this nonquadratic instance.
    assert results[1.0]["under"] > 0
    # Safety 1.25 is enough here (and on the broader 36-config sweep in the
    # report). If this fails on a new problem class, raise the factor or
    # fall back to the certified bound.
    assert results[1.25]["under"] == 0, (
        f"safety 1.25 under-estimated {results[1.25]['under']}/"
        f"{results[1.25]['n']} times on nonquadratic log-cosh; "
        f"min ratio {results[1.25]['min']:.3f}"
    )
