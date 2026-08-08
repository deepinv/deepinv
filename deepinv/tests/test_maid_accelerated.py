"""Tests for accelerated MAID (Zhang-Hager nonmonotone LS + BB step init)."""

from __future__ import annotations

import pytest
import torch

from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    QuadraticBilevelLS,
    SmoothHypergradientOracle,
    accelerated_maid_config,
)


def _rand_cond(n, d, cond, gen, dtype):
    G = torch.randn(n, d, generator=gen, dtype=dtype)
    Q, _ = torch.linalg.qr(G, mode="reduced")
    s = torch.logspace(0, torch.log10(torch.tensor(float(cond))), d, dtype=dtype)
    return Q * s


def _make_problem(cond=5.0, n=40, d=3, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)

    def mat():
        return _rand_cond(n, d, cond, gen, dtype)

    prob = QuadraticBilevelLS(
        mat(),
        mat(),
        mat(),
        torch.randn(n, generator=gen, dtype=dtype),
        torch.randn(n, generator=gen, dtype=dtype),
    )
    return prob


def _alpha0(prob):
    M = prob.A1 @ prob._P @ prob.A3
    Lf = float(2.0 * torch.linalg.eigvalsh(M.T @ M)[-1].item())
    return 1.0 / max(Lf, 1e-8)


def test_accelerated_config_defaults():
    cfg = accelerated_maid_config(max_iter=3, alpha0=0.1)
    assert cfg.nonmonotone is True
    assert cfg.bb_init is True
    assert cfg.max_iter == 3
    assert 0.0 <= cfg.eta_ref < 1.0


def test_config_rejects_bad_eta_ref():
    with pytest.raises(ValueError, match="eta_ref"):
        MAID(
            SmoothHypergradientOracle(_make_problem()),
            MAIDConfig(nonmonotone=True, eta_ref=1.0),
        )
    with pytest.raises(ValueError, match="eta_ref"):
        MAID(
            SmoothHypergradientOracle(_make_problem()),
            MAIDConfig(nonmonotone=True, eta_ref=-0.1),
        )


def test_config_rejects_bad_bb_form():
    with pytest.raises(ValueError, match="bb_form"):
        MAID(
            SmoothHypergradientOracle(_make_problem()),
            MAIDConfig(bb_init=True, bb_form="medium"),
        )


def test_vanilla_defaults_unchanged():
    """Accelerated switches are off by default (pure Algorithm 3.1)."""
    cfg = MAIDConfig()
    assert cfg.nonmonotone is False
    assert cfg.bb_init is False


def test_accelerated_decreases_f():
    prob = _make_problem(cond=8.0, seed=1)
    th0 = torch.ones(prob.d, dtype=torch.float64)
    f0 = float(prob.f_closed_form(th0).item())
    cfg = accelerated_maid_config(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=_alpha0(prob),
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=15,
        max_iter=10,
        tol=1e-8,
        g_convex=True,
        check_descent_direction=False,
    )
    th, hist = MAID(SmoothHypergradientOracle(prob), cfg).run(th0.clone())
    f_final = float(prob.f_closed_form(th).item())
    assert f_final < f0
    assert hist["n_upper_iters"] >= 1
    assert "C_ref" in hist
    assert "bb_used" in hist
    assert len(hist["bb_used"]) == hist["n_upper_iters"]


def test_nonmonotone_eta_ref_zero_matches_monotone_path_shape():
    """eta_ref=0 keeps a one-step memory; both paths must decrease f."""
    prob = _make_problem(cond=4.0, seed=2)
    th0 = torch.ones(prob.d, dtype=torch.float64)
    a0 = _alpha0(prob)
    base = dict(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=a0,
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=15,
        max_iter=6,
        tol=1e-8,
        g_convex=True,
        check_descent_direction=False,
        bb_init=False,
    )
    th_m, hist_m = MAID(
        SmoothHypergradientOracle(_make_problem(cond=4.0, seed=2)),
        MAIDConfig(nonmonotone=False, **base),
    ).run(th0.clone())
    th_n, hist_n = MAID(
        SmoothHypergradientOracle(_make_problem(cond=4.0, seed=2)),
        MAIDConfig(nonmonotone=True, eta_ref=0.0, **base),
    ).run(th0.clone())
    f0 = float(_make_problem(cond=4.0, seed=2).f_closed_form(th0).item())
    assert hist_m["f_exact"][-1] < f0
    assert hist_n["f_exact"][-1] < f0


def test_bb_fallback_when_sy_nonpositive():
    """Unit test of _bb_step fallback logic."""
    prob = _make_problem(cond=3.0, seed=3)
    maid = MAID(
        SmoothHypergradientOracle(prob),
        MAIDConfig(bb_init=True, bb_form="long", alpha_min=1e-8, alpha_max=1e2),
    )
    s = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    y = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)  # <s,y> < 0
    alpha, ok = maid._bb_step(s, y, fallback_alpha=0.05)
    assert ok is False
    assert alpha == pytest.approx(0.05)

    y_pos = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)  # <s,y> = 2
    alpha, ok = maid._bb_step(s, y_pos, fallback_alpha=0.05)
    assert ok is True
    # long: <s,s>/<s,y> = 1/2
    assert alpha == pytest.approx(0.5)


def test_bb_short_form():
    prob = _make_problem(cond=3.0, seed=4)
    maid = MAID(
        SmoothHypergradientOracle(prob),
        MAIDConfig(bb_init=True, bb_form="short", alpha_min=1e-8, alpha_max=1e2),
    )
    s = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    y = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    # short: <s,y>/<y,y> = 1/1 = 1
    alpha, ok = maid._bb_step(s, y, fallback_alpha=0.01)
    assert ok is True
    assert alpha == pytest.approx(1.0)


def test_bb_clamp():
    prob = _make_problem(cond=3.0, seed=5)
    maid = MAID(
        SmoothHypergradientOracle(prob),
        MAIDConfig(bb_init=True, bb_form="long", alpha_min=0.1, alpha_max=0.2),
    )
    s = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
    y = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    # long would be 100, clamp to 0.2
    alpha, ok = maid._bb_step(s, y, fallback_alpha=0.05)
    assert ok is True
    assert alpha == pytest.approx(0.2)


def test_accelerated_on_cheap_problem_reduces_or_keeps_bt():
    """On a cheap (cond=2) problem, accelerated should not explode BT count."""
    prob = _make_problem(cond=2.0, n=50, d=4, seed=6)
    th0 = torch.ones(prob.d, dtype=torch.float64)
    a0 = _alpha0(prob)
    common = dict(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=a0,
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=15,
        max_iter=15,
        tol=1e-8,
        g_convex=True,
        check_descent_direction=False,
    )
    _, hist_v = MAID(
        SmoothHypergradientOracle(_make_problem(cond=2.0, n=50, d=4, seed=6)),
        MAIDConfig(nonmonotone=False, bb_init=False, **common),
    ).run(th0.clone())
    _, hist_a = MAID(
        SmoothHypergradientOracle(_make_problem(cond=2.0, n=50, d=4, seed=6)),
        accelerated_maid_config(**common),
    ).run(th0.clone())
    # Both must finish; accelerated should not produce more failures than
    # a large multiple of vanilla (sanity, not a hard performance claim).
    assert hist_a["n_backtrack_failures"] <= max(5 * hist_v["n_backtrack_failures"], 20)
    assert hist_a["f_exact"][-1] <= hist_v["f_exact"][0] * 1.01
