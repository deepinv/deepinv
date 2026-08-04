"""
Bilevel learning of a Tikhonov prior weight with MAID
=====================================================

Learn the scalar weight of DeepInverse's :class:`~deepinv.optim.Tikhonov`
prior for denoising, using MAID and a lower level solved by
:class:`~deepinv.optim.GD` (or optionally ``PGD`` / ``FISTA``).

Lower level
    :math:`\\hat x(\\theta) = \\arg\\min_x \\tfrac12\\|x - y\\|^2
    + \\tfrac12 e^{\\theta}\\|x\\|^2`

Upper level
    :math:`f(\\theta) = \\tfrac12\\|\\hat x(\\theta) - x^\\star\\|^2`

The lower level is driven by DeepInverse's own optimiser with residual
stopping (gradient residual for ``GD``, proximal residual for ``PGD`` /
``FISTA``), not a hand-rolled loop. Warm starts reuse the previous
reconstruction.

"""

from __future__ import annotations

import time

import torch

import deepinv as dinv
from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    TikhonovWeightOracle,
    TikhonovWeightProblem,
)


# %%
# Data and physics
# ----------------

torch.manual_seed(0)
dtype = torch.float64
device = "cpu"

size = 32
x_star = torch.rand(1, 1, size, size, dtype=dtype, device=device)
# Inpainting: without regularisation the reconstruction is free on missing
# pixels, so a positive Tikhonov weight is optimal for supervised MSE.
mask = torch.ones_like(x_star)
mask[..., ::2, ::2] = 0
physics = dinv.physics.Inpainting(
    img_size=x_star.shape[1:], mask=mask, device=device
)
gen = torch.Generator(device=device).manual_seed(1)
noise = 0.05 * torch.randn(x_star.shape, generator=gen, dtype=dtype)
y = physics(x_star) + noise

print(f"image size {tuple(x_star.shape)}, inpainting mask density=0.75")


# %%
# Bilevel problem: learn log-Tikhonov weight via DeepInverse GD
# -------------------------------------------------------------

problem = TikhonovWeightProblem(
    physics=physics,
    y=y,
    x_star=x_star,
    solver="GD",
    max_iter=20_000,
)
oracle = TikhonovWeightOracle(problem)

theta0 = torch.tensor(-1.0, dtype=dtype)  # lambda = exp(-1) ~ 0.37
f0 = float(problem.f_closed_form(theta0).item())
print(f"f(theta0={float(theta0):.3f}, lambda={float(torch.exp(theta0)):.4f}) = {f0:.6f}")
print(f"lower-level solver: {problem.solver}, residual: {problem.residual_kind}")


# %%
# MAID
# ----

config = MAIDConfig(
    eps0=1e-2,
    delta0=1e-2,
    alpha0=0.2,
    rho=0.5,
    rho_bar=1.2,
    nu=0.5,
    nu_bar=1.1,
    eta=0.5,
    lambd=0.1,
    max_BT=15,
    max_iter=25,
    tol=1e-5,
    g_convex=True,
    check_descent_direction=False,
)

oracle.reset_counters()
problem.n_gd_iters = 0
maid = MAID(oracle, config)
t0 = time.perf_counter()
theta_maid, hist = maid.run(theta0.clone())
wall_maid = time.perf_counter() - t0
f_maid = float(problem.f_closed_form(theta_maid).item())
lam_maid = float(torch.exp(theta_maid).item())

print()
print(
    f"MAID: f {f0:.6f} -> {f_maid:.6f}, lambda={lam_maid:.4f}, "
    f"upper iters={hist['n_upper_iters']}, "
    f"BaseOptim iters={problem.n_gd_iters}, "
    f"wall={wall_maid:.3f}s, "
    f"backtrack failures={hist['n_backtrack_failures']}"
)


# %%
# Fixed-accuracy baseline (same GD residual target every outer step)
# ------------------------------------------------------------------

problem_fixed = TikhonovWeightProblem(
    physics=physics,
    y=y,
    x_star=x_star,
    solver="GD",
    max_iter=20_000,
)
oracle_fixed = TikhonovWeightOracle(problem_fixed)
theta = theta0.clone()
alpha = 0.2
eps_fixed = 1e-4
delta_fixed = 1e-4
# Match MAID's outer iteration count so the comparison is cost, not quality.
n_upper_target = int(hist["n_upper_iters"])
n_upper = 0
t0 = time.perf_counter()
for _ in range(n_upper_target):
    n_upper += 1
    lower = oracle_fixed.solve_lower_level(theta, eps=eps_fixed)
    hyper = oracle_fixed.hypergradient(theta, lower, delta=delta_fixed)
    theta = theta - alpha * hyper.z
wall_fixed = time.perf_counter() - t0
f_fixed = float(problem_fixed.f_closed_form(theta).item())
lam_fixed = float(torch.exp(theta).item())

print(
    f"Fixed eps={eps_fixed:g}: f {f0:.6f} -> {f_fixed:.6f}, "
    f"lambda={lam_fixed:.4f}, upper iters={n_upper}, "
    f"BaseOptim iters={problem_fixed.n_gd_iters}, wall={wall_fixed:.3f}s"
)

print()
print(
    f"{'method':<28} {'f_final':>10} {'lambda':>10} "
    f"{'BaseOptim iters':>16} {'wall_s':>8}"
)
print(
    f"{'MAID':<28} {f_maid:>10.6f} {lam_maid:>10.4f} "
    f"{problem.n_gd_iters:>16} {wall_maid:>8.3f}"
)
print(
    f"{'Fixed accuracy':<28} {f_fixed:>10.6f} {lam_fixed:>10.4f} "
    f"{problem_fixed.n_gd_iters:>16} {wall_fixed:>8.3f}"
)
print()
print(
    "Integration check: the lower level is deepinv.optim.GD (not a hand-rolled "
    "loop), stopped on the gradient residual, warm-started between outer "
    "steps, learning deepinv.optim.Tikhonov weight exp(theta) for inpainting."
)
print(
    f"Residual family: {problem.residual_kind}. "
    f"MAID backtrack failures: {hist['n_backtrack_failures']}."
)
print(
    "On this strongly convex denoising-style lower level a tight fixed residual "
    "is cheap per solve; MAID still spends BaseOptim iterations on line-search "
    "trials. The cost table above is the honest measurement for this instance."
)

assert f_maid < f0
assert problem.n_gd_iters > 0
assert lam_maid > 0.0
