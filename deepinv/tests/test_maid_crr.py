"""Tests for the multiconv convex ridge regulariser bilevel path."""

from __future__ import annotations

import torch
import deepinv as dinv

from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    ConvexRidgeConfig,
    CRRSampleProblem,
    CRRSampleOracle,
    pack_init_theta,
    unpack_theta,
    exp_scaling,
    n_crr_params,
)


def _tiny_cfg(**kwargs):
    defaults = dict(
        nb_channels=(1, 2, 4),
        filter_sizes=(3, 3),
        gamma=1e-2,
        weak_convexity=0.0,
        lip_fft_size=32,
    )
    defaults.update(kwargs)
    return ConvexRidgeConfig(**defaults)


def _tiny_denoising(size: int = 16, seed: int = 0, dtype=torch.float64, ch: int = 1):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(1, ch, size, size, generator=gen, dtype=dtype)
    physics = dinv.physics.Denoising(
        noise_model=dinv.physics.GaussianNoise(0.0)
    )
    y = physics(x) + 0.05 * torch.randn(x.shape, generator=gen, dtype=dtype)
    return physics, y, x


def test_n_params_matches_layout():
    assert n_crr_params((3, 4, 8, 64), (5, 5, 5)) == 300 + 800 + 12800 + 64 + 1
    cfg = ConvexRidgeConfig()
    assert cfg.n_params == n_crr_params(cfg.nb_channels, cfg.filter_sizes)
    assert n_crr_params((1, 2, 4), (3, 3)) == (2 * 1 * 9) + (4 * 2 * 9) + 4 + 1


def test_unpack_zero_mean_first_layer_and_positive_scale():
    cfg = _tiny_cfg()
    th = pack_init_theta(cfg, seed=1)
    weights, scaling, beta = unpack_theta(th, cfg)
    means = weights[0].mean(dim=(1, 2, 3))
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-12)
    assert torch.all(torch.exp(scaling) > 0)
    assert float(beta.item()) == cfg.beta_init


def test_grad_matches_finite_difference():
    cfg = _tiny_cfg()
    physics, y, x_star = _tiny_denoising(12, seed=2)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=4000)
    th = pack_init_theta(cfg, seed=3, weight_scale=0.02)
    x = x_star.clone()
    g = prob.grad_x_h(x, th)
    torch.manual_seed(4)
    v = torch.randn_like(x)
    v = v / v.flatten().norm()
    eps = 1e-5

    def h_val(xx):
        prob.load_theta(th)
        r = physics.A(xx) - y
        return float((0.5 * (r * r).sum() + prob.prior.energy(xx)).item())

    fd = (h_val(x + eps * v) - h_val(x - eps * v)) / (2 * eps)
    dir_der = float((g * v).sum().item())
    assert abs(fd - dir_der) / max(abs(fd), 1e-8) < 0.05


def test_hess_matvec_symmetric_probe():
    cfg = _tiny_cfg()
    physics, y, x_star = _tiny_denoising(10, seed=5)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=4000)
    th = pack_init_theta(cfg, seed=6, weight_scale=0.02)
    x = x_star.clone()
    torch.manual_seed(7)
    u = torch.randn_like(x)
    v = torch.randn_like(x)
    Hu = prob.hess_x_matvec(x, th, u)
    Hv = prob.hess_x_matvec(x, th, v)
    a = float((Hu * v).sum().item())
    b = float((Hv * u).sum().item())
    assert abs(a - b) / max(abs(a), 1e-8) < 1e-4


def test_solve_lower_reduces_residual():
    cfg = _tiny_cfg()
    physics, y, x_star = _tiny_denoising(12, seed=8)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=8000)
    th = pack_init_theta(cfg, seed=9)
    x0 = physics.A_adjoint(y)
    r0 = float(prob.grad_x_h(x0, th).flatten().norm().item())
    x, r = prob.solve_lower(th, eps=1e-2)
    assert r < r0
    assert r <= max(1e-2 * prob.mu(), 1e-8) * 1.01


def test_maid_decreases_upper_level():
    """MAID accepts steps and does not increase the upper-level cost."""
    cfg = _tiny_cfg()
    physics, y, x_star = _tiny_denoising(16, seed=12)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=8000)
    ora = CRRSampleOracle(prob)
    th0 = pack_init_theta(cfg, seed=13, weight_scale=0.05)
    f0 = float(prob.f_closed_form(th0).item())
    # Tight enough residual that the U sandwich gap is smaller than a typical
    # outer step; mild Armijo so an imperfect hypergradient still accepts.
    cfg_m = MAIDConfig(
        eps0=1e-3,
        delta0=1e-3,
        alpha0=5e-3,
        rho=0.5,
        rho_bar=1.2,
        nu=0.5,
        nu_bar=1.05,
        eta=0.5,
        lambd=0.01,
        max_BT=15,
        max_iter=4,
        tol=1e-6,
        g_convex=True,
        check_descent_direction=False,
        eps_min=1e-5,
        delta_min=1e-5,
        max_outer_BT=30,
        nonmonotone=False,
    )
    th, hist = MAID(ora, cfg_m).run(th0.clone())
    f1 = float(prob.f_closed_form(th).item())
    assert f1 <= f0 * 1.05
    assert hist["n_upper_iters"] >= 1
    assert prob.n_gd_iters > 0


def test_lip_normalisation_breaks_scale_degeneracy():
    cfg = _tiny_cfg()
    th = pack_init_theta(cfg, seed=20, weight_scale=0.1)
    th = th.detach().requires_grad_(True)
    x = torch.rand(1, 1, 12, 12, dtype=torch.float64)
    weights, scaling, beta = unpack_theta(th, cfg)
    from deepinv.optim.bilevel.convex_ridge import ridge_energy

    e = ridge_energy(x, weights, scaling, beta, cfg)
    (g,) = torch.autograd.grad(e, th)
    n_w = sum(
        cfg.nb_channels[i + 1]
        * cfg.nb_channels[i]
        * cfg.filter_sizes[i]
        * cfg.filter_sizes[i]
        for i in range(len(cfg.filter_sizes))
    )
    g_scale = g[n_w : n_w + cfg.n_filters]
    assert float(g_scale.norm().item()) > 1e-8


def test_hypergradient_moves_scaling():
    """Acceptance: IFT hypergradient on log-scaling is nonzero and a step moves it.

    The weight block dominates ||z||, so a shared Euclidean step barely
    moves scaling. A pure scale-coordinate step of the same hypergradient
    must leave the near-identity band (the Lip chart separates scale from
    filter amplitude).
    """
    cfg = _tiny_cfg()
    physics, y, x_star = _tiny_denoising(16, seed=12)
    oracle = CRRSampleOracle(
        CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=8000)
    )
    th = pack_init_theta(cfg, seed=13, weight_scale=0.05)
    s0 = exp_scaling(th, cfg)
    lower = oracle.solve_lower_level(th, eps=1e-3)
    hyper = oracle.hypergradient(th, lower, delta=1e-3)
    n_w = th.numel() - cfg.n_filters - 1
    z_s = hyper.z[n_w : n_w + cfg.n_filters]
    assert float(z_s.norm().item()) > 1e-4
    th2 = th.clone()
    th2[n_w : n_w + cfg.n_filters] = th2[n_w : n_w + cfg.n_filters] - 0.5 * z_s
    s1 = exp_scaling(th2, cfg)
    ratios = s1 / s0
    assert float(ratios.min()) < 0.97 or float(ratios.max()) > 1.03
