"""Tests for the convex ridge regulariser bilevel path."""

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
    renormalise_free_kernels,
    n_crr_params,
)


def _tiny_denoising(size: int = 16, seed: int = 0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 1, size, size, generator=gen, dtype=dtype)
    physics = dinv.physics.Denoising(
        noise_model=dinv.physics.GaussianNoise(0.0)
    )
    y = physics(x) + 0.05 * torch.randn(
        x.shape, generator=gen, dtype=dtype
    )
    return physics, y, x


def test_n_params_default_is_208():
    assert n_crr_params(8, 5, 1) == 208
    assert ConvexRidgeConfig().n_params == 208


def test_unpack_exp_positive_lambdas_and_unit_norm():
    cfg = ConvexRidgeConfig(n_kernels=4, kernel_size=3)
    th = pack_init_theta(cfg, seed=1, log_lambda0=-2.0)
    kernels, lambdas = unpack_theta(th, cfg)
    assert torch.all(lambdas > 0)
    means = kernels.mean(dim=(-2, -1))
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-12)
    norms = kernels.flatten(1).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)
    n_w = cfg.n_kernels * cfg.n_channels * cfg.kernel_size * cfg.kernel_size
    free = th[:n_w].reshape(
        cfg.n_kernels, cfg.n_channels, cfg.kernel_size, cfg.kernel_size
    )
    free_norms = free.flatten(1).norm(dim=1)
    assert torch.allclose(
        free_norms,
        torch.full_like(free_norms, cfg.free_kernel_scale),
        atol=1e-10,
    )


def test_unit_norm_breaks_scale_degeneracy_for_lambda_grad():
    """With unit-norm kernels, d energy / d log lambda is not identically zero."""
    cfg = ConvexRidgeConfig(n_kernels=2, kernel_size=3, gamma=1e-2)
    th = pack_init_theta(cfg, seed=20, log_lambda0=-1.0)
    th = th.detach().requires_grad_(True)
    x = torch.rand(1, 1, 12, 12, dtype=torch.float64)
    kernels, lambdas = unpack_theta(th, cfg)
    from deepinv.optim.bilevel.convex_ridge import ridge_energy

    e = ridge_energy(x, kernels, lambdas, cfg)
    (g,) = torch.autograd.grad(e, th)
    n_w = cfg.n_kernels * cfg.n_channels * cfg.kernel_size * cfg.kernel_size
    g_loglam = g[n_w:]
    assert float(g_loglam.norm().item()) > 1e-6


def test_renormalise_preserves_forward_kernels_and_lambdas():
    cfg = ConvexRidgeConfig(n_kernels=3, kernel_size=3, free_kernel_scale=1.0)
    th = pack_init_theta(cfg, seed=7)
    n_w = cfg.n_kernels * cfg.n_channels * cfg.kernel_size * cfg.kernel_size
    th_stretch = th.clone()
    th_stretch[:n_w] = th_stretch[:n_w] * 5.0
    k0, lam0 = unpack_theta(th_stretch, cfg)
    th_chart = renormalise_free_kernels(th_stretch, cfg)
    k1, lam1 = unpack_theta(th_chart, cfg)
    assert torch.allclose(k0, k1, atol=1e-12)
    assert torch.allclose(lam0, lam1, atol=1e-12)
    free = th_chart[:n_w].reshape(
        cfg.n_kernels, cfg.n_channels, cfg.kernel_size, cfg.kernel_size
    )
    norms = free.flatten(1).norm(dim=1)
    assert torch.allclose(
        norms, torch.full_like(norms, cfg.free_kernel_scale), atol=1e-10
    )


def test_grad_matches_finite_difference():
    cfg = ConvexRidgeConfig(n_kernels=2, kernel_size=3, gamma=1e-2)
    physics, y, x_star = _tiny_denoising(12, seed=2)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=2000)
    th = pack_init_theta(cfg, seed=3)
    x = x_star.clone()
    g = prob.grad_x_h(x, th)
    torch.manual_seed(4)
    v = torch.randn_like(x)
    v = v / v.flatten().norm()
    eps = 1e-5

    def h_val(xx):
        prob.load_theta(th)
        r = physics.A(xx) - y
        return float(
            (0.5 * (r * r).sum() + prob.prior.energy(xx)).item()
        )

    fd = (h_val(x + eps * v) - h_val(x - eps * v)) / (2 * eps)
    dir_der = float((g * v).sum().item())
    assert abs(fd - dir_der) / max(abs(fd), 1e-8) < 0.05


def test_hess_matvec_symmetric_probe():
    cfg = ConvexRidgeConfig(n_kernels=2, kernel_size=3, gamma=1e-2)
    physics, y, x_star = _tiny_denoising(10, seed=5)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=2000)
    th = pack_init_theta(cfg, seed=6)
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
    cfg = ConvexRidgeConfig(n_kernels=2, kernel_size=3, gamma=1e-2)
    physics, y, x_star = _tiny_denoising(12, seed=8)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=8000)
    th = pack_init_theta(cfg, seed=9)
    x0 = physics.A_adjoint(y)
    r0 = float(prob.grad_x_h(x0, th).flatten().norm().item())
    x, r = prob.solve_lower(th, eps=1e-2)
    assert r < r0
    assert r <= max(1e-2 * prob.mu(), 1e-8) * 1.01


def test_maid_decreases_upper_level():
    cfg = ConvexRidgeConfig(n_kernels=2, kernel_size=3, gamma=1e-2)
    physics, y, x_star = _tiny_denoising(12, seed=10)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=8000)
    ora = CRRSampleOracle(prob)
    th0 = pack_init_theta(cfg, seed=11, log_lambda0=-0.5)
    f0 = float(prob.f_closed_form(th0).item())
    cfg_m = MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=1e-3,
        rho=0.5,
        rho_bar=1.2,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=10,
        max_iter=5,
        tol=1e-6,
        g_convex=True,
        check_descent_direction=False,
        eps_min=1e-4,
        delta_min=1e-4,
    )
    th, hist = MAID(ora, cfg_m).run(th0.clone())
    f1 = float(prob.f_closed_form(th).item())
    assert f1 <= f0 * 1.05
    assert hist["n_upper_iters"] >= 1
    assert prob.n_gd_iters > 0


def test_maid_moves_lambdas_under_unit_norm_chart():
    """Acceptance: with unit-norm kernels and free scale O(1), lambdas move."""
    cfg = ConvexRidgeConfig(
        n_kernels=4, kernel_size=3, gamma=1e-2, free_kernel_scale=1.0
    )
    physics, y, x_star = _tiny_denoising(16, seed=12)
    oracle = CRRSampleOracle(
        CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=8000)
    )
    th0 = pack_init_theta(cfg, seed=13, log_lambda0=-1.0)
    _, lam0 = unpack_theta(th0, cfg)
    cfg_m = MAIDConfig(
        eps0=5e-2,
        delta0=5e-2,
        alpha0=2e-2,
        rho=0.5,
        rho_bar=1.2,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=10,
        max_iter=6,
        tol=1e-5,
        g_convex=True,
        check_descent_direction=False,
        eps_min=1e-5,
        delta_min=1e-5,
    )
    th, _ = MAID(oracle, cfg_m).run(th0.clone())
    th = renormalise_free_kernels(th, cfg)
    _, lam1 = unpack_theta(th, cfg)
    ratios = lam1 / lam0
    assert float(ratios.min()) < 0.95 or float(ratios.max()) > 1.05
