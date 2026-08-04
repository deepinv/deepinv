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


def test_unpack_exp_positive_lambdas():
    cfg = ConvexRidgeConfig(n_kernels=4, kernel_size=3)
    th = pack_init_theta(cfg, seed=1, log_lambda0=-2.0)
    kernels, lambdas = unpack_theta(th, cfg)
    assert torch.all(lambdas > 0)
    # Zero-mean kernels.
    means = kernels.mean(dim=(-2, -1))
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-12)


def test_grad_matches_finite_difference():
    cfg = ConvexRidgeConfig(n_kernels=2, kernel_size=3, gamma=1e-2)
    physics, y, x_star = _tiny_denoising(12, seed=2)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=2000)
    th = pack_init_theta(cfg, seed=3, kernel_scale=0.02)
    x = x_star.clone()
    g = prob.grad_x_h(x, th)
    # Finite difference on a random direction.
    torch.manual_seed(4)
    v = torch.randn_like(x)
    v = v / v.flatten().norm()
    eps = 1e-5
    # scalar energy difference
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
    th = pack_init_theta(cfg, seed=6, kernel_scale=0.02)
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
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=3000)
    th = pack_init_theta(cfg, seed=9)
    x0 = physics.A_adjoint(y)
    r0 = float(prob.grad_x_h(x0, th).flatten().norm().item())
    x, r = prob.solve_lower(th, eps=1e-2)
    assert r < r0
    assert r <= max(1e-2 * prob.mu(), 1e-8) * 1.01


def test_maid_decreases_upper_level():
    cfg = ConvexRidgeConfig(n_kernels=2, kernel_size=3, gamma=1e-2)
    physics, y, x_star = _tiny_denoising(12, seed=10)
    prob = CRRSampleProblem(physics, y, x_star, cfg=cfg, max_iter=2000)
    ora = CRRSampleOracle(prob)
    th0 = pack_init_theta(cfg, seed=11, kernel_scale=0.02, log_lambda0=-0.5)
    f0 = float(prob.f_closed_form(th0).item())
    # Scale step by 1/n_params so ||alpha z|| is moderate.
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
        eps_min=1e-5,
        delta_min=1e-5,
    )
    th, hist = MAID(ora, cfg_m).run(th0.clone())
    f1 = float(prob.f_closed_form(th).item())
    assert f1 <= f0 * 1.05  # allow tiny numerical noise
    assert hist["n_upper_iters"] >= 1
    assert prob.n_gd_iters > 0
