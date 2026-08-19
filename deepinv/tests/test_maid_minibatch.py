"""Tests for the MAID minibatch extension.

Requirements checked (supervisor order):

1. Determinism: same data, seed, chunk schedule -> bitwise identical z.
2. Chunk invariance: fixed reduction tree => bitwise identical z across
   chunk sizes (not a tolerance claim).
3. Error-bound aggregation: omega_mean = (1/m) sum omega_i, and it
   dominates ||z - grad f||.
4. Trajectory invariance: MAID with chunk_size 4 and 64 (or 1 and m)
   follows the same f / theta path.
5. Peak working memory scales with chunk size, not with dataset size m.
6. Goal-oriented estimator is per-sample; Krylov recycling does not span
   samples.
"""

from __future__ import annotations

import tracemalloc

import pytest
import torch

from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
)
from deepinv.optim.bilevel.minibatch import (
    MinibatchOracle,
    make_quadratic_dataset,
    mean_error_bound,
    wrap_smooth_dataset,
)


def _exact_mean_hypergradient(problems, theta: torch.Tensor) -> torch.Tensor:
    acc = None
    for p in problems:
        zi = p.exact_hypergradient(theta)
        acc = zi.clone() if acc is None else acc + zi
    return acc / float(len(problems))


def _exact_mean_f(problems, theta: torch.Tensor) -> float:
    total = 0.0
    for p in problems:
        total += float(p.f_closed_form(theta).item())
    return total / float(len(problems))


# ---------------------------------------------------------------------------
# 3. Error-bound aggregation (pure function + oracle)
# ---------------------------------------------------------------------------


def test_mean_error_bound_is_average():
    assert mean_error_bound([1.0, 3.0, 5.0]) == pytest.approx(3.0)
    assert mean_error_bound([2.0]) == pytest.approx(2.0)


def test_mean_error_bound_rejects_empty():
    with pytest.raises(ValueError):
        mean_error_bound([])


def test_aggregated_omega_dominates_true_mean_error():
    """omega_mean = (1/m) sum omega_i >= ||z_mean - grad f||."""
    m, cond, d = 6, 5.0, 3
    problems = make_quadratic_dataset(m, cond=cond, n=30, d=d, seed=2)
    oracles = wrap_smooth_dataset(problems)
    mb = MinibatchOracle(oracles, chunk_size=2)
    theta = torch.ones(d, dtype=torch.float64)
    eps, delta = 1e-3, 1e-3

    lower = mb.solve_lower_level(theta, eps=eps)
    hyper = mb.hypergradient(theta, lower, delta=delta)
    omega = mb.error_bound(theta, lower, hyper, eps=eps, delta=delta)

    z_exact = _exact_mean_hypergradient(problems, theta)
    true_err = float((hyper.z - z_exact).norm().item())
    assert omega >= true_err - 1e-12
    # Aggregation formula.
    assert omega == pytest.approx(mean_error_bound(mb.last_omega_parts))
    assert len(mb.last_omega_parts) == m


def test_wrong_aggregation_would_fail_when_omegas_differ():
    """Document why the mean (not max, not first) is required.

    If one sample has a much larger omega_i, using omega_0 alone under-bounds
    the mean error. Using max is valid but looser; the contract is the mean.
    """
    omegas = [1e-6, 1e-6, 1.0]
    mean_w = mean_error_bound(omegas)
    assert mean_w == pytest.approx((1e-6 + 1e-6 + 1.0) / 3.0)
    assert mean_w < max(omegas)
    assert mean_w > omegas[0]


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def test_hypergradient_bitwise_deterministic_across_runs():
    problems = make_quadratic_dataset(8, cond=10.0, n=30, d=3, seed=0)
    theta = torch.linspace(0.5, 1.5, 3, dtype=torch.float64)
    eps, delta = 1e-3, 1e-3

    def once():
        # Fresh oracles so counters do not share state; same problem objects.
        mb = MinibatchOracle(wrap_smooth_dataset(problems), chunk_size=3)
        lower = mb.solve_lower_level(theta, eps=eps)
        hyper = mb.hypergradient(theta, lower, delta=delta)
        return hyper.z.clone(), float(mb.error_bound(theta, lower, hyper, eps, delta))

    z1, w1 = once()
    z2, w2 = once()
    assert torch.equal(z1, z2)
    assert w1 == w2


# ---------------------------------------------------------------------------
# 2. Chunk invariance (bitwise, fixed reduction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_a,chunk_b", [(1, 4), (2, 8), (1, 8)])
def test_chunk_size_bitwise_invariance(chunk_a, chunk_b):
    m = 8
    problems = make_quadratic_dataset(m, cond=8.0, n=30, d=3, seed=1)
    theta = torch.ones(3, dtype=torch.float64)
    eps, delta = 5e-4, 5e-4

    def run(cs):
        mb = MinibatchOracle(wrap_smooth_dataset(problems), chunk_size=cs)
        lower = mb.solve_lower_level(theta, eps=eps)
        hyper = mb.hypergradient(theta, lower, delta=delta)
        omega = mb.error_bound(theta, lower, hyper, eps=eps, delta=delta)
        return hyper.z.clone(), omega, mb.g(lower.x).clone()

    z_a, w_a, g_a = run(chunk_a)
    z_b, w_b, g_b = run(chunk_b)
    assert torch.equal(z_a, z_b)
    assert w_a == w_b
    assert torch.equal(g_a, g_b)


# ---------------------------------------------------------------------------
# 4. Trajectory invariance under chunk size
# ---------------------------------------------------------------------------


def test_maid_trajectory_invariant_to_chunk_size():
    """chunk_size 1 vs m: same f path and final theta (bitwise)."""
    m, cond, d = 4, 12.0, 3
    # Expensive enough lower level that MAID does real work, but small m
    # so the test stays fast.
    problems = make_quadratic_dataset(m, cond=cond, n=40, d=d, seed=3)
    theta0 = torch.ones(d, dtype=torch.float64)

    def Lf_mean():
        # Use first sample's upper Lip as a conservative step-size scale.
        p0 = problems[0]
        M = p0.A1 @ p0._P @ p0.A3
        return float(2.0 * torch.linalg.eigvalsh(M.T @ M)[-1].item())

    alpha0 = 1.0 / max(Lf_mean(), 1e-8)
    cfg = MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=alpha0,
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=10,
        max_iter=5,
        tol=1e-8,
        g_convex=True,
        check_descent_direction=False,
    )

    def run(cs):
        mb = MinibatchOracle(wrap_smooth_dataset(problems), chunk_size=cs)
        th, hist = MAID(mb, cfg).run(theta0.clone())
        return th, list(hist["f_exact"]), list(hist["z_norm"]), list(hist["alpha"])

    th1, f1, z1, a1 = run(1)
    thm, fm, zm, am = run(m)
    assert torch.equal(th1, thm)
    assert f1 == fm
    assert z1 == zm
    assert a1 == am


# ---------------------------------------------------------------------------
# 5. Peak memory bounded by chunk, not dataset
# ---------------------------------------------------------------------------


def test_peak_working_bytes_scales_with_chunk_not_m():
    """Peak working bytes (instrumented) grow with chunk_size, not m."""
    d, n, cond = 8, 50, 5.0
    # Large m, small chunks vs large chunks.
    m = 16
    problems = make_quadratic_dataset(m, cond=cond, n=n, d=d, seed=4)
    theta = torch.ones(d, dtype=torch.float64)

    peaks = {}
    for cs in (1, 2, 4, 8):
        mb = MinibatchOracle(wrap_smooth_dataset(problems), chunk_size=cs)
        lower = mb.solve_lower_level(theta, eps=1e-3)
        mb.hypergradient(theta, lower, delta=1e-3)
        peaks[cs] = mb.peak_working_bytes

    # Monotone non-decreasing in chunk size.
    assert peaks[1] <= peaks[2] <= peaks[4] <= peaks[8]
    # Roughly linear: chunk 8 holds about 8x the per-sample working set of
    # chunk 1 (allow slack for stack metadata).
    assert peaks[8] >= 4 * peaks[1]

    # Flat in m at fixed chunk_size: peak for m=8 and m=16 with cs=2 comparable.
    problems_small = make_quadratic_dataset(8, cond=cond, n=n, d=d, seed=4)
    mb_s = MinibatchOracle(wrap_smooth_dataset(problems_small), chunk_size=2)
    lower = mb_s.solve_lower_level(theta, eps=1e-3)
    mb_s.hypergradient(theta, lower, delta=1e-3)
    # Same chunk_size -> same peak working (per-chunk tensors only).
    assert mb_s.peak_working_bytes == peaks[2]


def test_tracemalloc_peak_not_linear_in_m_at_fixed_chunk():
    """tracemalloc peak for a hypergradient pass grows much less than m."""
    d, n, cond, cs = 6, 40, 4.0, 2
    theta = torch.ones(d, dtype=torch.float64)

    def peak_for(m):
        problems = make_quadratic_dataset(m, cond=cond, n=n, d=d, seed=5)
        mb = MinibatchOracle(wrap_smooth_dataset(problems), chunk_size=cs)
        tracemalloc.start()
        lower = mb.solve_lower_level(theta, eps=1e-3)
        mb.hypergradient(theta, lower, delta=1e-3)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    p8 = peak_for(8)
    p32 = peak_for(32)
    # If peak scaled fully with m, p32/p8 would be about 4. Allow growth
    # for the O(m) warm-start list but require sub-linear: less than 3x
    # when m grows 4x (working set dominated by the chunk).
    assert p32 < 3.0 * p8


# ---------------------------------------------------------------------------
# 6. Goal-oriented per chunk / per sample; recycling stays per-sample
# ---------------------------------------------------------------------------


def test_goal_oriented_minibatch_runs_and_omega_positive():
    m, d = 4, 3
    problems = make_quadratic_dataset(m, cond=6.0, n=25, d=d, seed=6)
    oracles = wrap_smooth_dataset(problems, goal_oriented=True, recycle_krylov=True)
    mb = MinibatchOracle(oracles, chunk_size=2)
    # Non-certified.
    assert mb.certified is False
    theta = torch.ones(d, dtype=torch.float64)
    lower = mb.solve_lower_level(theta, eps=1e-3)
    hyper = mb.hypergradient(theta, lower, delta=1e-3)
    omega = mb.error_bound(theta, lower, hyper, eps=1e-3, delta=1e-3)
    assert omega > 0.0
    z_exact = _exact_mean_hypergradient(problems, theta)
    # Safety factor 1.25 should not under-estimate on quadratic samples.
    true_err = float((hyper.z - z_exact).norm().item())
    assert omega >= true_err - 1e-10


def test_krylov_recycle_is_per_sample_not_cross_sample():
    """Recycling on vs off changes per-sample cost but not z bitwise when
    the main CG residual is driven to the same delta (directions only affect
    DWR, not z). Document: recycling cannot span samples.
    """
    m, d = 3, 3
    problems = make_quadratic_dataset(m, cond=5.0, n=25, d=d, seed=7)
    theta = torch.ones(d, dtype=torch.float64)
    eps, delta = 1e-3, 1e-3

    def run(recycle):
        oracles = wrap_smooth_dataset(
            problems, goal_oriented=True, recycle_krylov=recycle, cg_budget=5
        )
        mb = MinibatchOracle(oracles, chunk_size=1)
        lower = mb.solve_lower_level(theta, eps=eps)
        hyper = mb.hypergradient(theta, lower, delta=delta)
        omega = mb.error_bound(theta, lower, hyper, eps=eps, delta=delta)
        return hyper.z.clone(), omega

    z_on, w_on = run(True)
    z_off, w_off = run(False)
    # z comes from the main IFT solve, independent of DWR recycling.
    assert torch.equal(z_on, z_off)
    # omega may differ slightly with recycling quality; both must be positive.
    assert w_on > 0.0 and w_off > 0.0


def test_dwr_cost_scales_with_m_not_chunk_size():
    """Sample hypergradient count is m per outer hypergradient, any chunk."""
    m, d = 6, 3
    problems = make_quadratic_dataset(m, cond=4.0, n=20, d=d, seed=8)
    theta = torch.ones(d, dtype=torch.float64)
    for cs in (1, 2, 3, 6):
        mb = MinibatchOracle(
            wrap_smooth_dataset(problems, goal_oriented=True), chunk_size=cs
        )
        lower = mb.solve_lower_level(theta, eps=1e-3)
        mb.hypergradient(theta, lower, delta=1e-3)
        assert mb.n_sample_hypergradients == m
        assert mb.n_hypergradients == 1


# ---------------------------------------------------------------------------
# Integration: MAID decreases mean f on expensive multi-sample problem
# ---------------------------------------------------------------------------


def test_maid_minibatch_decreases_mean_f_expensive():
    """Build on the expensive end of the crossover (cond >= 20)."""
    m, cond, d = 4, 20.0, 3
    problems = make_quadratic_dataset(m, cond=cond, n=40, d=d, seed=9)
    theta0 = torch.ones(d, dtype=torch.float64)
    f0 = _exact_mean_f(problems, theta0)

    p0 = problems[0]
    M = p0.A1 @ p0._P @ p0.A3
    Lf = float(2.0 * torch.linalg.eigvalsh(M.T @ M)[-1].item())
    alpha0 = 1.0 / max(Lf, 1e-8)

    mb = MinibatchOracle(wrap_smooth_dataset(problems), chunk_size=2)
    cfg = MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=alpha0,
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=12,
        max_iter=8,
        tol=1e-8,
        g_convex=True,
        check_descent_direction=False,
    )
    th, hist = MAID(mb, cfg).run(theta0.clone())
    f_final = _exact_mean_f(problems, th)
    assert f_final < f0
    assert hist["n_upper_iters"] >= 1
    assert mb.n_gd_iters > 0


def test_uncertified_minibatch_requires_opt_in():
    problems = make_quadratic_dataset(2, cond=3.0, n=15, d=2, seed=10)
    mb = MinibatchOracle(
        wrap_smooth_dataset(problems, goal_oriented=True), chunk_size=1
    )
    with pytest.raises(ValueError, match="not certified"):
        MAID(mb, MAIDConfig(max_iter=1))
    # Opt-in works.
    th, _ = MAID(
        mb, MAIDConfig(max_iter=2, alpha0=0.1, g_convex=True), allow_uncertified=True
    ).run(torch.ones(2, dtype=torch.float64))
    assert th.shape == (2,)
