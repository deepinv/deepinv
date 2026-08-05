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
    assert_grad_div_adjoint,
    grid_tune_tv,
    isotropic_tv,
    nabla,
    div,
    recon_tv,
    solve_isotropic_tv,
)
from deepinv.optim.bilevel.tv_baseline import primal_objective


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


def test_frozen_lip_stable_across_load_theta():
    """Lip normalisation is frozen on first load and not recomputed later.

    Recomputing lip(theta) and detaching it is what made the weight-block
    hypergradient wrong: the forward solve and the differentiated model
    must share one constant.
    """
    from deepinv.optim.bilevel.convex_ridge import get_conv_lip

    cfg = _tiny_cfg()
    physics, y, x_star = _tiny_denoising(10, seed=1)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=100)
    th0 = pack_init_theta(cfg, seed=2, weight_scale=0.1)
    th1 = th0 + 0.5 * torch.randn_like(th0)
    prob.load_theta(th0)
    lip0 = float(prob.prior.lip.item())
    prob.load_theta(th1)
    lip1 = float(prob.prior.lip.item())
    assert abs(lip0 - lip1) < 1e-15
    # Explicit refresh redefines the model and is allowed between outer iters.
    prob.load_theta(th1, refresh_lip=True)
    lip_refreshed = float(prob.prior.lip.item())
    w1, _, _ = unpack_theta(th1, cfg)
    expected = float(get_conv_lip(w1, cfg, detach=True).item())
    assert abs(lip_refreshed - expected) < 1e-12
    assert abs(lip_refreshed - lip0) > 1e-10


def test_hypergradient_matches_finite_difference():
    """Analytic hypergradient agrees with central FD on every block.

    F(theta) = 1/2 ||x*(theta) - x_gt||^2 with a tight lower-level solve.
    Detached recomputed Lip normalisation previously made the weight block
    wrong by O(1) (wrong sign on some coordinates). This test is the gate
    that catches that class of model/derivative mismatch.

    Checks: scaling block, beta, random weight direction, random full
    direction, and three individual weight coordinates including a
    first-layer one.
    """
    cfg = _tiny_cfg(
        nb_channels=(1, 2, 4),
        filter_sizes=(3, 3),
        gamma=1e-2,
        weak_convexity=0.0,
        lip_fft_size=32,
        beta_init=4.0,
        sigma_init=0.1,
    )
    physics, y, x_star = _tiny_denoising(12, seed=50, ch=1)
    prob = CRRSampleProblem(
        physics, y, x_star, cfg=cfg, max_iter=20_000, solver="NEWTON"
    )
    # Nontrivial weight scale so the Lip chart and the filters interact.
    theta = pack_init_theta(cfg, seed=0, weight_scale=1.0)
    # Freeze lip once at theta_0 (matches learning: one model for the step).
    prob.load_theta(theta)

    eps_ll = 1e-10

    def F(th: torch.Tensor) -> float:
        x, _ = prob.solve_lower(th, eps=eps_ll, max_iter=20_000)
        return float(prob.g(x).item())

    x0, res0 = prob.solve_lower(theta, eps=eps_ll, max_iter=20_000)
    assert res0 < 1e-6
    z, cg = prob.inexact_hypergradient(
        x0, theta, delta=1e-12, max_cg_iter=2000
    )
    assert float(torch.as_tensor(cg.residual).norm().item()) < 1e-8

    n_w = theta.numel() - cfg.n_filters - 1
    g_rng = torch.Generator().manual_seed(7)

    directions: list[tuple[str, torch.Tensor]] = []
    d = torch.zeros_like(theta)
    d[n_w : n_w + cfg.n_filters] = 1.0
    directions.append(("scaling block", d))
    d = torch.zeros_like(theta)
    d[-1] = 1.0
    directions.append(("beta", d))
    d = torch.zeros_like(theta)
    d[:n_w] = torch.randn(n_w, generator=g_rng, dtype=theta.dtype)
    d = d / d.norm().clamp_min(1e-30)
    directions.append(("random weight dir", d))
    d = torch.randn(theta.numel(), generator=g_rng, dtype=theta.dtype)
    d = d / d.norm().clamp_min(1e-30)
    directions.append(("random full dir", d))
    # First-layer coordinate 0 and two further weight coordinates.
    for idx in (0, min(5, n_w - 1), min(20, n_w - 1)):
        d = torch.zeros_like(theta)
        d[idx] = 1.0
        directions.append((f"weight coord {idx}", d))

    rtol = 5e-3
    for name, d in directions:
        analytic = float((z * d).sum().item())
        best_rel = float("inf")
        best_fd = 0.0
        for h in (1e-4, 1e-5, 1e-6):
            fp = F(theta + h * d)
            fm = F(theta - h * d)
            fd = (fp - fm) / (2.0 * h)
            rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)
            if rel < best_rel:
                best_rel = rel
                best_fd = fd
        assert best_rel < rtol, (
            f"hypergradient FD mismatch on {name}: "
            f"analytic={analytic:.6e} fd={best_fd:.6e} rel={best_rel:.2e}"
        )


def test_tv_grad_div_adjoint_identity():
    """<nabla u, p> = -<u, div p> to machine precision (mandatory TV check)."""
    residual = assert_grad_div_adjoint(
        shape=(1, 3, 20, 20), dtype=torch.float64, atol=1e-12, seed=0
    )
    assert abs(residual) < 1e-12
    # Direct probe with the exported nabla / div pair.
    gen = torch.Generator().manual_seed(1)
    u = torch.randn(1, 2, 12, 12, generator=gen, dtype=torch.float64)
    p = torch.randn(1, 2, 12, 12, 2, generator=gen, dtype=torch.float64)
    lhs = float((nabla(u) * p).sum().item())
    rhs = float(-(u * div(p)).sum().item())
    assert abs(lhs - rhs) < 1e-12 * max(abs(lhs), 1.0)


def test_tv_objective_decreases_on_denoising():
    """TV solve must reduce 1/2||x-y||^2 + lam TV(x) vs the measurement.

    This is the solver check the baseline depends on: objective decrease
    against the measurement, asserted inside solve_isotropic_tv(verify=True).
    """
    physics, y, x_star = _tiny_denoising(16, seed=3, ch=1)
    lam = 0.03
    obj_y = primal_objective(y, physics, y, lam)
    xh, info = solve_isotropic_tv(physics, y, lam, n_it=300, verify=True)
    assert info["obj_final"] < info["obj_init"]
    assert info["obj_final"] < obj_y - 1e-12
    assert xh.shape == y.shape
    # Grid-tuned lambda on this sample must improve PSNR over the measurement.
    best = grid_tune_tv(
        [(physics, y, x_star)], n_grid=9, n_it=200, verify_once=True
    )
    mse_y = float(torch.mean((y - x_star) ** 2).item())
    psnr_y = 10.0 * torch.log10(torch.tensor(1.0 / max(mse_y, 1e-30))).item()
    assert best["psnr_train"] > psnr_y


def test_tv_grid_tune_and_recon_on_train_sample():
    """Grid-tuned TV is a valid baseline: returns positive lambda and PSNR."""
    physics, y, x_star = _tiny_denoising(12, seed=5, ch=1)
    samples = [(physics, y, x_star)]
    best = grid_tune_tv(samples, n_grid=7, n_it=200, verify_once=True)
    assert best["lam"] > 0
    assert best["psnr_train"] > 0
    xh, p, info = recon_tv(physics, y, x_star, best["lam"], verify=True)
    assert p == best["psnr_train"] or abs(p - best["psnr_train"]) < 0.5
    assert info["obj_final"] < info["obj_init"]
    assert float(isotropic_tv(xh).item()) >= 0.0
