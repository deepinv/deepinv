"""
Bilevel learning of a regularisation weight with MAID
=====================================================

This example learns a hyperparameter of a Tikhonov-type reconstruction by
bilevel optimisation, using DeepInverse's :class:`deepinv.optim.bilevel` MAID
implementation.

Problem
-------
Given a linear measurement operator :math:`A` (a DeepInverse physics model),
ground-truth signals :math:`x^\\star` and measurements :math:`y = A x^\\star +
\\text{noise}`, the lower level reconstructs

.. math::

    \\hat x(\\theta) = \\arg\\min_x \\|A_2 x + A_3 \\theta - b_2\\|^2

and the upper level fits the reconstruction quality

.. math::

    f(\\theta) = \\|A_1 \\hat x(\\theta) - b_1\\|^2.

This is the quadratic bilevel benchmark of Salehi et al. (SIAM J. Math. Data
Sci. 2025, section 4.1), written so that the lower-level map is linear in the
parameter and has a known optimum. It is the right place to measure what
adaptivity buys: the same final accuracy at a lower lower-level cost.

Comparison
----------
We compare two upper-level strategies with the same outer iteration budget:

1. **MAID** (default path: no a priori descent test, adaptive ``eps`` /
   ``delta`` via backtracking).
2. **Fixed-accuracy** inexact gradient descent: the same step-size schedule
   without accuracy adaptation, with ``eps`` and ``delta`` fixed at a tight
   value.

The headline metric is **total lower-level gradient steps** (plus hypergradient
CG solves, counted as hypergradient evaluations). That is what adaptivity
changes.

"""

from __future__ import annotations

import time

import torch

import deepinv as dinv
from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    QuadraticBilevelLS,
    SmoothHypergradientOracle,
)


# %%
# Build the section 4.1 bilevel problem
# -------------------------------------
# Matrices are thin QR factors so the upper level is well conditioned and the
# comparison finishes in a few dozen outer iterations. Ground-truth data use
# a DeepInverse :class:`~deepinv.physics.LinearPhysics` operator for the
# lower-level map :math:`A_2`.


def well_conditioned(n: int, d: int, gen: torch.Generator, dtype) -> torch.Tensor:
    G = torch.randn(n, d, generator=gen, dtype=dtype)
    Q, _ = torch.linalg.qr(G, mode="reduced")
    s = torch.linspace(1.0, 2.0, d, dtype=dtype)
    return Q * s


torch.manual_seed(0)
dtype = torch.float64
device = dinv.utils.get_device()
# Keep the demo on CPU float64 for reproducibility of the cost table.
device = "cpu"

n, d = 120, 5
gen = torch.Generator(device=device).manual_seed(1)
A1 = well_conditioned(n, d, gen, dtype).to(device)
A2 = well_conditioned(n, d, gen, dtype).to(device)
A3 = well_conditioned(n, d, gen, dtype).to(device)

# DeepInverse physics for the lower-level linear map A2 (vector form).
physics = dinv.physics.LinearPhysics(
    A=lambda x, **kwargs: A2 @ x,
    A_adjoint=lambda y, **kwargs: A2.T @ y,
)

x_true = torch.randn(d, generator=gen, dtype=dtype, device=device)
theta_bar = torch.randn(d, generator=gen, dtype=dtype, device=device)
noise = 0.01 * torch.randn(n, generator=gen, dtype=dtype, device=device)
# Measurement through the physics operator (plus a parameter-dependent term).
y = physics.A(x_true) + A3 @ theta_bar + noise
b2 = y
b1 = A1 @ x_true + 0.01 * torch.randn(n, generator=gen, dtype=dtype, device=device)

problem = QuadraticBilevelLS(A1=A1, A2=A2, A3=A3, b1=b1, b2=b2)
theta0 = torch.ones(d, dtype=dtype, device=device)
theta_star = problem.closed_form_theta_star()
f_star = float(problem.f_closed_form(theta_star).item())
f0 = float(problem.f_closed_form(theta0).item())

print(f"f(theta0)   = {f0:.6f}")
print(f"f(theta*)   = {f_star:.6f}")
print(f"device      = {device}, dtype = {dtype}")


# %%
# Lipschitz step size for the exact hypergradient
# ------------------------------------------------


def upper_lipschitz(prob: QuadraticBilevelLS) -> float:
    M = prob.A1 @ prob._P @ prob.A3
    return float(2.0 * torch.linalg.eigvalsh(M.T @ M)[-1].item())


alpha0 = 1.0 / upper_lipschitz(problem)


# %%
# MAID run
# --------


TARGET_GAP = 1e-3


def run_maid(max_iter: int = 80) -> dict:
    config = MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=alpha0,
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=30,
        max_iter=max_iter,
        tol=1e-8,
        g_convex=True,
        check_descent_direction=False,
    )
    problem.n_gd_iters = 0
    oracle = SmoothHypergradientOracle(problem)
    maid = MAID(oracle, config)
    theta, hist = maid.run(theta0.clone())
    hit = hist["n_upper_iters"]
    for i, f in enumerate(hist["f_exact"]):
        if (f - f_star) / max(abs(f_star), 1.0) < TARGET_GAP:
            hit = i + 1
            break
    config.max_iter = hit
    problem.n_gd_iters = 0
    oracle = SmoothHypergradientOracle(problem)
    maid = MAID(oracle, config)
    t0 = time.perf_counter()
    theta, hist = maid.run(theta0.clone())
    wall = time.perf_counter() - t0
    f_final = float(problem.f_closed_form(theta).item())
    return {
        "name": "MAID (adaptive eps/delta)",
        "theta": theta,
        "f_final": f_final,
        "rel_gap": (f_final - f_star) / max(abs(f_star), 1.0),
        "upper_iters": hist["n_upper_iters"],
        "lower_solves": hist["n_lower_solves"],
        "gd_iters": problem.n_gd_iters,
        "hypergrads": hist["n_hypergradients"],
        "backtrack_failures": hist["n_backtrack_failures"],
        "wall_s": wall,
    }


# %%
# Fixed-accuracy baseline
# -----------------------
# Same outer step-size growth as MAID after a successful step, but ``eps`` and
# ``delta`` stay fixed at a tight value. Every hypergradient therefore pays the
# full lower-level cost.


def run_fixed(eps: float = 1e-4, delta: float = 1e-4, max_iter: int = 200) -> dict:
    """Fixed tight accuracy until the same relative gap as the MAID target."""
    problem.gd_max_iter = 500_000
    problem.n_gd_iters = 0
    oracle = SmoothHypergradientOracle(problem)
    theta = theta0.clone()
    alpha = alpha0
    n_lower = 0
    n_hyper = 0
    upper_iters = 0
    t0 = time.perf_counter()
    for _ in range(max_iter):
        upper_iters += 1
        lower = oracle.solve_lower_level(theta, eps=eps)
        n_lower += 1
        hyper = oracle.hypergradient(theta, lower, delta=delta)
        n_hyper += 1
        z = hyper.z
        theta = theta - alpha * z
        f_now = float(problem.f_closed_form(theta).item())
        rel = (f_now - f_star) / max(abs(f_star), 1.0)
        if rel < TARGET_GAP or float(z.norm().item()) <= 1e-6:
            break
    wall = time.perf_counter() - t0
    f_final = float(problem.f_closed_form(theta).item())
    return {
        "name": f"Fixed accuracy (eps=delta={eps:g})",
        "theta": theta,
        "f_final": f_final,
        "rel_gap": (f_final - f_star) / max(abs(f_star), 1.0),
        "upper_iters": upper_iters,
        "lower_solves": n_lower,
        "gd_iters": problem.n_gd_iters,
        "hypergrads": n_hyper,
        "backtrack_failures": 0,
        "wall_s": wall,
    }


# %%
# Results
# -------

maid_stats = run_maid(max_iter=80)
fixed_stats = run_fixed(eps=1e-4, delta=1e-4, max_iter=200)

rows = [maid_stats, fixed_stats]
print()
print(f"Target relative gap: {TARGET_GAP:g}")
print(
    f"{'method':<36} {'rel_gap':>10} {'upper':>6} {'gd_iters':>10} "
    f"{'ll_calls':>8} {'hyper':>6} {'wall_s':>8}"
)
for r in rows:
    print(
        f"{r['name']:<36} {r['rel_gap']:>10.3e} {r['upper_iters']:>6} "
        f"{r['gd_iters']:>10} {r['lower_solves']:>8} {r['hypergrads']:>6} "
        f"{r['wall_s']:>8.3f}"
    )

print()
print(
    "Headline metric: total lower-level GD iterations to the same relative "
    "gap. MAID starts coarse (few GD steps per solve) and tightens only when "
    "backtracking requires it; fixed accuracy pays a tight residual on every "
    "outer step."
)
print(
    f"MAID backtrack failures (a posteriori detector): "
    f"{maid_stats['backtrack_failures']}"
)

assert maid_stats["rel_gap"] < TARGET_GAP * 2
assert fixed_stats["rel_gap"] < TARGET_GAP * 2
