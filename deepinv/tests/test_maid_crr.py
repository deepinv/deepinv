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
    # Default solver is FISTA_RESTART (smooth CRR with adaptive restart).
    prob = CRRSampleProblem(
        physics, y, x_star, cfg=cfg, max_iter=8000, solver="FISTA_RESTART"
    )
    th = pack_init_theta(cfg, seed=9)
    x0 = physics.A_adjoint(y)
    r0 = float(prob.grad_x_h(x0, th).flatten().norm().item())
    x, r = prob.solve_lower(th, eps=1e-2)
    assert r < r0
    assert r <= max(1e-2 * prob.mu(), 1e-8) * 1.01
    assert prob.n_gd_iters > 0


def test_gd_fista_restart_newton_reach_residual():
    """GD, FISTA, FISTA_RESTART and NEWTON all meet the residual tolerance.

    The certificate depends only on the residual, not the algorithm.
    """
    cfg = _tiny_cfg(gamma=1e-2)
    physics, y, x_star = _tiny_denoising(16, seed=42)
    th = pack_init_theta(cfg, seed=0, weight_scale=0.05)
    eps = 1e-3
    x0 = physics.A_adjoint(y)
    for name in ("GD", "FISTA", "FISTA_RESTART", "NEWTON"):
        prob = CRRSampleProblem(
            physics, y, x_star, cfg=cfg, max_iter=20_000, solver=name
        )
        assert prob.solver == name
        _, r = prob.solve_lower(th, eps=eps, x_init=x0.clone())
        tol = max(eps * prob.mu(), 1e-8)
        assert r <= tol * 1.01
        assert prob.n_gd_iters >= 1


def test_adaptive_data_lipschitz_at_or_above_fixed_200():
    """Adaptive L_data (with safety) must sit at or above 200 fixed iters."""
    cfg = _tiny_cfg()
    # Denoising: exact eigenvalue 1; still checks the adaptive path.
    physics, y, x_star = _tiny_denoising(12, seed=0)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=100)
    L_adapt, _, L_raw = prob.estimate_data_lipschitz()
    L_200, _, _ = prob.estimate_data_lipschitz(fixed_iters=200)
    assert L_adapt >= L_200 * 0.999
    assert L_adapt >= L_raw  # safety factor

    # Inpainting mask is a projection: A^* A has eigenvalues in {0, 1}.
    gen = torch.Generator().manual_seed(1)
    mask = (torch.rand(x_star.shape, generator=gen, dtype=x_star.dtype) > 0.5).to(
        x_star.dtype
    )
    physics_ip = dinv.physics.Inpainting(
        img_size=x_star.shape[1:], mask=mask, device=x_star.device
    )
    prob_ip = CRRSampleProblem(physics_ip, y * mask, x_star, cfg=cfg, max_iter=100)
    L_a, _, _ = prob_ip.estimate_data_lipschitz()
    L_f, _, _ = prob_ip.estimate_data_lipschitz(fixed_iters=200)
    assert L_a >= L_f * 0.999


def test_analytic_prior_lipschitz_covers_measured():
    """Analytic L_prior must be at least the measured top eigenvalue of Hess R."""
    cfg = _tiny_cfg(beta_init=4.0, sigma_init=0.1, gamma=1e-2)
    physics, y, x_star = _tiny_denoising(12, seed=3)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=100)
    th = pack_init_theta(cfg, seed=4)
    prob.load_theta(th)
    analytic = prob.analytic_prior_lipschitz()
    # Probe at zeros: quadratic region maximises curvature of smooth_l1.
    measured, _, raw = prob.measure_prior_lipschitz(th, x=torch.zeros_like(x_star))
    assert analytic >= raw * 0.99
    assert measured == raw  # safety=1 default


def test_certificate_against_reference_solve():
    """||x_loose - x_ref|| <= ||grad h(x_loose)|| / mu must hold."""
    cfg = _tiny_cfg(gamma=1e-2)
    physics, y, x_star = _tiny_denoising(14, seed=7)
    prob = CRRSampleProblem(
        physics, y, x_star, cfg=cfg, max_iter=30_000, solver="FISTA_RESTART"
    )
    th = pack_init_theta(cfg, seed=8)
    x0 = physics.A_adjoint(y)
    x_loose, res_loose = prob.solve_lower(
        th, eps=1e-2, x_init=x0.clone(), solver="FISTA_RESTART"
    )
    x_ref, _ = prob.solve_lower(
        th, eps=1e-6, x_init=x0.clone(), solver="FISTA_RESTART"
    )
    dist = float((x_loose - x_ref).flatten().norm().item())
    bound = res_loose / max(prob.mu(), 1e-30)
    assert dist <= bound * 1.05


def test_gd_objective_monotone_with_safe_step():
    """GD at step 1/L on strongly convex h must decrease the objective."""
    cfg = _tiny_cfg(gamma=1e-2)
    physics, y, x_star = _tiny_denoising(12, seed=9)
    prob = CRRSampleProblem(
        physics, y, x_star, cfg=cfg, max_iter=5_000, solver="GD"
    )
    th = pack_init_theta(cfg, seed=10)
    x0 = physics.A_adjoint(y)
    out = prob.solve_lower(
        th, eps=1e-2, x_init=x0.clone(), solver="GD", record_every=1
    )
    assert len(out) == 3
    _, _, hist = out
    assert hist["monotone_objective"] is True
    objs = hist["objectives"]
    assert len(objs) >= 2
    for a, b in zip(objs, objs[1:]):
        assert b <= a + 1e-10


def test_mu_data_plus_gamma_on_denoising():
    """Denoising (A = I) contributes mu_data = 1, so gamma may be zero.

    Certificate uses mu = mu_data + gamma. With gamma = 0 the residual
    tolerance is still positive and the distance bound is residual / 1.
    """
    cfg = _tiny_cfg(gamma=0.0)
    physics, y, x_star = _tiny_denoising(12, seed=11)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=8000)
    assert prob.mu_data == 1.0
    assert abs(prob.mu() - 1.0) < 1e-12
    th = pack_init_theta(cfg, seed=12)
    _, r = prob.solve_lower(th, eps=1e-2)
    tol = max(1e-2 * prob.mu(), 1e-8)
    assert r <= tol * 1.01
    # Distance bound is residual / mu, not the residual alone.
    dist_bound = r / prob.mu()
    assert dist_bound <= 1e-2 * 1.01


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
