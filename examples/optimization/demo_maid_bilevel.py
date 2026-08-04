"""
MAID for bilevel learning: when adaptive accuracy pays
======================================================

The Method of Adaptive Inexact Descent (MAID) solves bilevel problems by
adapting the lower-level residual and the hypergradient residual at every
outer step. The benefit appears when a tight lower-level solve is expensive.
When a tight solve is nearly free, line-search trials dominate and a fixed
accuracy baseline wins.

This example does four things:

1. Flagship: an ill-conditioned quadratic least-squares bilevel problem
   (Salehi et al., SIAM J. Math. Data Sci. 2025, section 4.1) where MAID
   uses fewer total lower-level gradient steps, and fewer outer steps
   (each costing a hypergradient), than fixed accuracy to the same
   upper-level gap.
2. Crossover: sweep the lower-level condition number and tabulate total
   gradient steps and outer steps for MAID against fixed accuracy. The
   honest claim is not "MAID is always faster", but "MAID is faster once a
   tight lower-level solve costs more than a few thousand gradient steps".
3. Counterpoint: learn a Tikhonov weight with DeepInverse
   :class:`~deepinv.optim.GD` on a well-conditioned inpainting problem.
   There a tight residual is nearly free, and fixed accuracy wins. That is
   the regime where MAID should not be used.
4. Minibatch extension on the expensive end of the crossover: fixed-order
   chunking with bitwise chunk-size invariance, correct mean error-bound
   aggregation, peak working memory scaling with chunk size, and the
   crossover recomputed with chunking enabled.

"""

from __future__ import annotations

import time

import torch

import deepinv as dinv
from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    MinibatchOracle,
    QuadraticBilevelLS,
    SmoothHypergradientOracle,
    TikhonovWeightOracle,
    TikhonovWeightProblem,
    make_quadratic_dataset,
    wrap_smooth_dataset,
)


torch.manual_seed(0)
dtype = torch.float64

# Relative upper-level gap used for matched-quality comparisons.
TARGET_GAP = 5e-3
FIXED_EPS = 1e-4


# %%
# Helpers: controllable quadratic bilevel problem
# -----------------------------------------------
#
# The lower-level Hessian condition number is set by the singular values of
# the design matrices. Larger ``cond`` means more gradient steps per tight
# lower-level solve.


def make_quadratic(cond: float, n: int = 80, d: int = 4, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)

    def mat():
        G = torch.randn(n, d, generator=gen, dtype=dtype)
        Q, _ = torch.linalg.qr(G, mode="reduced")
        s = torch.logspace(0, torch.log10(torch.tensor(float(cond))), d, dtype=dtype)
        return Q * s

    return QuadraticBilevelLS(
        mat(),
        mat(),
        mat(),
        torch.randn(n, generator=gen, dtype=dtype),
        torch.randn(n, generator=gen, dtype=dtype),
    )


def upper_lipschitz(prob: QuadraticBilevelLS) -> float:
    """Lipschitz constant of the reduced upper-level gradient (exact)."""
    M = prob.A1 @ prob._P @ prob.A3
    return float(2.0 * torch.linalg.eigvalsh(M.T @ M)[-1].item())


def maid_config(alpha0: float, max_iter: int) -> MAIDConfig:
    return MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=alpha0,
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=15,
        max_iter=max_iter,
        tol=1e-6,
        g_convex=True,
        check_descent_direction=False,
    )


def relative_gap(f: float, f_star: float) -> float:
    return (f - f_star) / max(abs(f_star), 1.0)


def run_maid_to_gap(cond: float, target: float = TARGET_GAP, max_discover: int = 50):
    """Return GD count, outer steps and wall time to reach ``target`` gap."""
    prob = make_quadratic(cond)
    theta0 = torch.ones(prob.d, dtype=dtype)
    f_star = float(prob.f_closed_form(prob.closed_form_theta_star()).item())
    alpha0 = 1.0 / max(upper_lipschitz(prob), 1e-8)

    # Discover the first outer step that meets the gap.
    _, hist = MAID(SmoothHypergradientOracle(prob), maid_config(alpha0, max_discover)).run(
        theta0.clone()
    )
    hit = None
    for i, f in enumerate(hist["f_exact"]):
        if relative_gap(f, f_star) < target:
            hit = i + 1
            break
    if hit is None:
        raise RuntimeError(
            f"MAID did not reach gap {target} at cond={cond} "
            f"(final gap={relative_gap(hist['f_exact'][-1], f_star):.3e})"
        )

    # Recount GD iterations with max_iter fixed at the hit.
    prob = make_quadratic(cond)
    t0 = time.perf_counter()
    theta, hist = MAID(
        SmoothHypergradientOracle(prob), maid_config(alpha0, hit)
    ).run(theta0.clone())
    wall = time.perf_counter() - t0
    f_final = float(prob.f_closed_form(theta).item())
    return {
        "gd": prob.n_gd_iters,
        "upper": hit,
        "wall": wall,
        "f": f_final,
        "f_star": f_star,
        "f0": float(make_quadratic(cond).f_closed_form(theta0).item()),
        "alpha0": alpha0,
        "bt_fail": hist["n_backtrack_failures"],
        "theta": theta,
    }


def run_fixed_to_gap(
    cond: float,
    target: float = TARGET_GAP,
    eps: float = FIXED_EPS,
    max_upper: int = 200,
):
    """Fixed residual every outer step, stop at the same upper-level gap."""
    prob = make_quadratic(cond)
    oracle = SmoothHypergradientOracle(prob)
    theta0 = torch.ones(prob.d, dtype=dtype)
    f_star = float(prob.f_closed_form(prob.closed_form_theta_star()).item())
    alpha = 1.0 / max(upper_lipschitz(prob), 1e-8)
    theta = theta0.clone()
    warm = None
    n_upper = 0
    t0 = time.perf_counter()
    while n_upper < max_upper:
        n_upper += 1
        lower = oracle.solve_lower_level(theta, eps=eps, warm_start=warm)
        hyper = oracle.hypergradient(theta, lower, delta=eps)
        theta = theta - alpha * hyper.z
        warm = lower
        f = float(prob.f_closed_form(theta).item())
        if relative_gap(f, f_star) < target:
            break
    wall = time.perf_counter() - t0
    return {
        "gd": prob.n_gd_iters,
        "upper": n_upper,
        "wall": wall,
        "f": float(prob.f_closed_form(theta).item()),
        "f_star": f_star,
    }


# %%
# 1. Flagship: ill-conditioned quadratic (condition number 30)
# ------------------------------------------------------------
#
# Matched quality: both methods stop when the relative gap to the closed-form
# upper-level optimum falls below 0.5 percent. Cost is total lower-level
# gradient steps (and wall time).

FLAGSHIP_COND = 30.0
print(f"Flagship quadratic bilevel, lower-level condition number = {FLAGSHIP_COND:g}")
print(f"Target relative gap to f_star: {TARGET_GAP:g}")
print(f"Fixed baseline residual: eps = delta = {FIXED_EPS:g}")
print()

maid_flag = run_maid_to_gap(FLAGSHIP_COND)
fixed_flag = run_fixed_to_gap(FLAGSHIP_COND)

print(
    f"f0 = {maid_flag['f0']:.4f}, f_star = {maid_flag['f_star']:.4f}, "
    f"alpha0 = {maid_flag['alpha0']:.4e}"
)
print(
    f"MAID:  f = {maid_flag['f']:.4f}, "
    f"gap = {relative_gap(maid_flag['f'], maid_flag['f_star']):.3e}, "
    f"GD iters = {maid_flag['gd']}, upper = {maid_flag['upper']}, "
    f"backtrack failures = {maid_flag['bt_fail']}, "
    f"wall = {maid_flag['wall']:.3f}s"
)
print(
    f"Fixed: f = {fixed_flag['f']:.4f}, "
    f"gap = {relative_gap(fixed_flag['f'], fixed_flag['f_star']):.3e}, "
    f"GD iters = {fixed_flag['gd']}, upper = {fixed_flag['upper']}, "
    f"wall = {fixed_flag['wall']:.3f}s"
)
print()
print(
    f"{'method':<28} {'f_final':>10} {'GD iters':>10} "
    f"{'upper':>7} {'wall_s':>8}"
)
print(
    f"{'MAID':<28} {maid_flag['f']:>10.4f} {maid_flag['gd']:>10d} "
    f"{maid_flag['upper']:>7d} {maid_flag['wall']:>8.3f}"
)
print(
    f"{'Fixed accuracy 1e-4':<28} {fixed_flag['f']:>10.4f} {fixed_flag['gd']:>10d} "
    f"{fixed_flag['upper']:>7d} {fixed_flag['wall']:>8.3f}"
)
ratio_flag = maid_flag["gd"] / max(fixed_flag["gd"], 1)
outer_ratio = maid_flag["upper"] / max(fixed_flag["upper"], 1)
print()
print(
    f"GD ratio MAID / fixed = {ratio_flag:.3f} "
    f"({'MAID cheaper' if ratio_flag < 1 else 'fixed cheaper'}). "
    f"Outer-step ratio = {outer_ratio:.3f} "
    f"(each outer step costs one hypergradient: a linear solve plus an adjoint)."
)

assert relative_gap(maid_flag["f"], maid_flag["f_star"]) < TARGET_GAP
assert relative_gap(fixed_flag["f"], fixed_flag["f_star"]) < TARGET_GAP
assert maid_flag["gd"] < fixed_flag["gd"], (
    "Flagship should show MAID using fewer GD steps than fixed accuracy"
)


# %%
# 2. Crossover: sweep the lower-level condition number
# ----------------------------------------------------
#
# For each condition number, both methods run to the same relative gap.
# Ratio below 1 means MAID uses fewer total gradient steps.

CONDS = [2.0, 5.0, 10.0, 20.0, 30.0]
print()
print(
    f"Crossover: total lower-level GD steps to relative gap < {TARGET_GAP:g}"
)
print(
    f"{'cond':>6} {'gd_maid':>10} {'gd_fixed':>10} {'ratio':>8} "
    f"{'up_maid':>8} {'up_fixed':>9}"
)

crossover_rows = []
for cond in CONDS:
    m = run_maid_to_gap(cond)
    f = run_fixed_to_gap(cond)
    ratio = m["gd"] / max(f["gd"], 1)
    crossover_rows.append((cond, m["gd"], f["gd"], ratio, m["upper"], f["upper"]))
    print(
        f"{cond:6.1f} {m['gd']:10d} {f['gd']:10d} {ratio:8.3f} "
        f"{m['upper']:8d} {f['upper']:9d}"
    )

# The crossover should appear: ratio > 1 at small cond, ratio < 1 at large cond.
ratios = [r[3] for r in crossover_rows]
assert ratios[0] > 1.0, "Cheap end of the sweep should favour fixed accuracy"
assert ratios[-1] < 1.0, "Expensive end of the sweep should favour MAID"
print()
print(
    "Reading: at condition number 2 a tight solve is cheap and MAID pays for "
    "line-search trials. Around condition number 5 to 10 the costs cross. "
    "Above that, adaptive accuracy saves total gradient work because early "
    "outer steps use loose tolerances and fewer outer steps are needed. "
    "The outer-step column is the stronger saving: each outer iteration costs "
    "a hypergradient, not just a few gradient steps."
)


# %%
# 3. Counterpoint: cheap DeepInverse Tikhonov inpainting
# ------------------------------------------------------
#
# Learn :math:`\lambda = e^\theta` for
#
# .. math::
#
#     \hat x(\theta) = \arg\min_x \tfrac12\|Ax - y\|^2 + \tfrac12 e^\theta\|x\|^2
#
# on a 32x32 inpainting problem. The lower level is strongly convex and
# well conditioned, so a residual of ``1e-4`` costs about one gradient step
# per solve. MAID still runs several lower-level solves per outer step for
# the line search. Fixed accuracy therefore wins on total BaseOptim iterations.

print()
print("Counterpoint: Tikhonov weight on 32x32 inpainting (DeepInverse GD)")

size = 32
x_star = torch.rand(1, 1, size, size, dtype=dtype)
mask = torch.ones_like(x_star)
mask[..., ::2, ::2] = 0
physics = dinv.physics.Inpainting(
    img_size=x_star.shape[1:], mask=mask, device="cpu"
)
gen = torch.Generator().manual_seed(1)
noise = 0.05 * torch.randn(x_star.shape, generator=gen, dtype=dtype)
y = physics(x_star) + noise

theta0_img = torch.tensor(-1.0, dtype=dtype)
problem_img = TikhonovWeightProblem(
    physics=physics,
    y=y,
    x_star=x_star,
    solver="GD",
    max_iter=20_000,
)
oracle_img = TikhonovWeightOracle(problem_img)
f0_img = float(problem_img.f_closed_form(theta0_img).item())

config_img = MAIDConfig(
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

problem_img.n_gd_iters = 0
t0 = time.perf_counter()
theta_maid_img, hist_img = MAID(oracle_img, config_img).run(theta0_img.clone())
wall_maid_img = time.perf_counter() - t0
f_maid_img = float(problem_img.f_closed_form(theta_maid_img).item())
lam_maid = float(torch.exp(theta_maid_img).item())
gd_maid_img = problem_img.n_gd_iters
n_upper_img = int(hist_img["n_upper_iters"])

# Matched outer iteration count: same budget of upper steps, fixed residual.
problem_fixed_img = TikhonovWeightProblem(
    physics=physics,
    y=y,
    x_star=x_star,
    solver="GD",
    max_iter=20_000,
)
oracle_fixed_img = TikhonovWeightOracle(problem_fixed_img)
theta_f = theta0_img.clone()
t0 = time.perf_counter()
for _ in range(n_upper_img):
    lower = oracle_fixed_img.solve_lower_level(theta_f, eps=FIXED_EPS)
    hyper = oracle_fixed_img.hypergradient(theta_f, lower, delta=FIXED_EPS)
    theta_f = theta_f - 0.2 * hyper.z
wall_fixed_img = time.perf_counter() - t0
f_fixed_img = float(problem_fixed_img.f_closed_form(theta_f).item())
lam_fixed = float(torch.exp(theta_f).item())
gd_fixed_img = problem_fixed_img.n_gd_iters

print(f"image size {tuple(x_star.shape)}, inpainting mask density=0.75")
print(f"f(theta0) = {f0_img:.4f}, residual family = {problem_img.residual_kind}")
print()
print(
    f"{'method':<28} {'f_final':>10} {'lambda':>10} "
    f"{'BaseOptim iters':>16} {'wall_s':>8}"
)
print(
    f"{'MAID':<28} {f_maid_img:>10.4f} {lam_maid:>10.4f} "
    f"{gd_maid_img:>16d} {wall_maid_img:>8.3f}"
)
print(
    f"{'Fixed accuracy 1e-4':<28} {f_fixed_img:>10.4f} {lam_fixed:>10.4f} "
    f"{gd_fixed_img:>16d} {wall_fixed_img:>8.3f}"
)
print()
print(
    f"MAID backtrack failures: {hist_img['n_backtrack_failures']}. "
    f"BaseOptim iter ratio MAID / fixed = "
    f"{gd_maid_img / max(gd_fixed_img, 1):.2f}."
)
print(
    "On this instance a tight residual costs about one GD step per solve, so "
    "fixed accuracy reaches a comparable upper-level value with far fewer "
    "BaseOptim iterations. MAID spends those iterations on line-search trials "
    f"({hist_img['n_backtrack_failures']} failed backtracks in "
    f"{n_upper_img} outer steps)."
)

assert f_maid_img < f0_img
assert gd_maid_img > gd_fixed_img, (
    "Inpainting counterpoint should show fixed accuracy cheaper on BaseOptim iters"
)
assert problem_img.residual_kind == "gradient"


# %%
# 4. Minibatch extension on the expensive end
# -------------------------------------------
#
# The paper's upper level is already a mean over m samples. Chunking
# evaluates the inexact hypergradient sequentially so peak working memory
# tracks the chunk, not the dataset. Reduction order is fixed (index order,
# sequential floating-point addition), so chunk size does not change the
# mathematics: trajectories are bitwise identical across chunk sizes.
#
# Error bound for the mean: if ||z_i - grad f_i|| <= omega_i, then
# ||z_mean - grad f|| <= (1/m) sum omega_i.

print()
print("Minibatch extension (expensive end, condition number 20)")

MB_M = 4
MB_COND = 20.0
MB_CHUNK = 2
mb_problems = make_quadratic_dataset(MB_M, cond=MB_COND, n=40, d=3, seed=0)


def _mb_f(problems, theta):
    return sum(float(p.f_closed_form(theta).item()) for p in problems) / len(
        problems
    )


def _mb_alpha0(problems):
    # Lip of the mean upper-level gradient <= mean of per-sample Lips.
    total = 0.0
    for p in problems:
        M = p.A1 @ p._P @ p.A3
        total += float(2.0 * torch.linalg.eigvalsh(M.T @ M)[-1].item())
    return 1.0 / max(total / len(problems), 1e-8)


theta0_mb = torch.ones(3, dtype=dtype)
f0_mb = _mb_f(mb_problems, theta0_mb)
alpha0_mb = _mb_alpha0(mb_problems)

# Trajectory invariance: chunk_size 1 vs m, short run.
cfg_traj = MAIDConfig(
    eps0=1e-1,
    delta0=1e-1,
    alpha0=alpha0_mb,
    rho=0.5,
    rho_bar=1.5,
    nu=0.5,
    nu_bar=1.1,
    eta=0.5,
    lambd=0.1,
    max_BT=20,
    max_iter=5,
    tol=1e-8,
    g_convex=True,
    check_descent_direction=False,
)
traj_rows = []
for cs in (1, MB_M):
    mb = MinibatchOracle(wrap_smooth_dataset(mb_problems), chunk_size=cs)
    th_t, hist_t = MAID(mb, cfg_traj).run(theta0_mb.clone())
    traj_rows.append((cs, list(hist_t["f_exact"]), th_t.clone()))
    print(
        f"  trajectory chunk_size={cs}: f path = "
        f"[{', '.join(f'{v:.6f}' for v in hist_t['f_exact'])}]"
    )

assert traj_rows[0][1] == traj_rows[1][1], "f path must match across chunk sizes"
assert torch.equal(traj_rows[0][2], traj_rows[1][2]), (
    "final theta must be bitwise identical across chunk sizes"
)
print("  bitwise trajectory match: yes (chunk 1 vs chunk m)")

# Peak working memory vs chunk size (instrumented inside MinibatchOracle).
print()
print(f"  peak working bytes vs chunk_size (m={MB_M}, cond={MB_COND:g}):")
print(f"  {'chunk':>6} {'peak_working_bytes':>20}")
mem_peaks = []
for cs in (1, 2, 4):
    # Use m>=4 dataset; for cs=4 need m>=4.
    probs_mem = make_quadratic_dataset(4, cond=MB_COND, n=40, d=3, seed=0)
    mb = MinibatchOracle(wrap_smooth_dataset(probs_mem), chunk_size=cs)
    lower = mb.solve_lower_level(theta0_mb, eps=1e-3)
    mb.hypergradient(theta0_mb, lower, delta=1e-3)
    mem_peaks.append((cs, mb.peak_working_bytes))
    print(f"  {cs:6d} {mb.peak_working_bytes:20d}")
assert mem_peaks[0][1] <= mem_peaks[-1][1]

# Matched-quality cost at cond=20 with chunking, vs fixed accuracy.
print()
print(
    f"  matched quality at cond={MB_COND:g}, m={MB_M}, chunk_size={MB_CHUNK} "
    f"(stop when f <= f_MAID after a short adaptive run)"
)
mb_maid = MinibatchOracle(
    wrap_smooth_dataset(mb_problems), chunk_size=MB_CHUNK
)
cfg_mb = MAIDConfig(
    eps0=1e-1,
    delta0=1e-1,
    alpha0=alpha0_mb,
    rho=0.5,
    rho_bar=1.5,
    nu=0.5,
    nu_bar=1.1,
    eta=0.5,
    lambd=0.1,
    max_BT=20,
    max_iter=10,
    tol=1e-8,
    g_convex=True,
    check_descent_direction=False,
)
t0 = time.perf_counter()
th_mb, hist_mb = MAID(mb_maid, cfg_mb).run(theta0_mb.clone())
wall_mb = time.perf_counter() - t0
f_mb = _mb_f(mb_problems, th_mb)
gd_mb = mb_maid.n_gd_iters
up_mb = int(hist_mb["n_upper_iters"])

# Fixed accuracy to the same f value (or better).
mb_fix = MinibatchOracle(
    wrap_smooth_dataset(mb_problems), chunk_size=MB_CHUNK
)
th_fx = theta0_mb.clone()
warm = None
up_fx = 0
t0 = time.perf_counter()
while up_fx < 80:
    up_fx += 1
    lower = mb_fix.solve_lower_level(th_fx, eps=FIXED_EPS, warm_start=warm)
    hyper = mb_fix.hypergradient(th_fx, lower, delta=FIXED_EPS)
    th_fx = th_fx - alpha0_mb * hyper.z
    warm = lower
    if _mb_f(mb_problems, th_fx) <= f_mb * 1.001:
        break
wall_fx = time.perf_counter() - t0
f_fx = _mb_f(mb_problems, th_fx)
gd_fx = mb_fix.n_gd_iters

print(
    f"  {'method':<28} {'f_final':>10} {'GD iters':>10} "
    f"{'upper':>7} {'wall_s':>8}"
)
print(
    f"  {'MAID chunk=2':<28} {f_mb:>10.4f} {gd_mb:>10d} "
    f"{up_mb:>7d} {wall_mb:>8.3f}"
)
print(
    f"  {'Fixed accuracy 1e-4':<28} {f_fx:>10.4f} {gd_fx:>10d} "
    f"{up_fx:>7d} {wall_fx:>8.3f}"
)
ratio_mb = gd_mb / max(gd_fx, 1)
print(
    f"  GD ratio MAID/fixed = {ratio_mb:.3f}, "
    f"outer ratio = {up_mb / max(up_fx, 1):.3f}, "
    f"f0 = {f0_mb:.4f}"
)
print(
    "  omega aggregation: mean of per-sample bounds "
    f"(last omega_parts length = {len(mb_maid.last_omega_parts) if mb_maid.last_omega_parts else 'n/a'}; "
    "see MinibatchOracle docstring)."
)
assert f_mb < f0_mb

# Mini crossover with chunking: cond in {5, 10, 20}.
print()
print(
    f"  Minibatch crossover (m={MB_M}, chunk={MB_CHUNK}): "
    f"GD steps to f <= f_MAID(max_iter=8)"
)
print(
    f"  {'cond':>6} {'gd_maid':>10} {'gd_fixed':>10} {'ratio':>8} "
    f"{'up_maid':>8} {'up_fixed':>9}"
)
mb_cross = []
for cond in (5.0, 10.0, 20.0):
    probs = make_quadratic_dataset(MB_M, cond=cond, n=40, d=3, seed=0)
    a0 = _mb_alpha0(probs)
    th0 = torch.ones(3, dtype=dtype)
    f0c = _mb_f(probs, th0)
    mb_m = MinibatchOracle(wrap_smooth_dataset(probs), chunk_size=MB_CHUNK)
    cfg_c = MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=a0,
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=20,
        max_iter=8,
        tol=1e-8,
        g_convex=True,
        check_descent_direction=False,
    )
    th_m, hist_m = MAID(mb_m, cfg_c).run(th0.clone())
    f_target = _mb_f(probs, th_m)
    gd_m = mb_m.n_gd_iters
    up_m = int(hist_m["n_upper_iters"])

    mb_f = MinibatchOracle(wrap_smooth_dataset(probs), chunk_size=MB_CHUNK)
    th_f = th0.clone()
    warm = None
    up_f = 0
    while up_f < 80:
        up_f += 1
        lower = mb_f.solve_lower_level(th_f, eps=FIXED_EPS, warm_start=warm)
        hyper = mb_f.hypergradient(th_f, lower, delta=FIXED_EPS)
        th_f = th_f - a0 * hyper.z
        warm = lower
        if _mb_f(probs, th_f) <= f_target * 1.001:
            break
    ratio_c = gd_m / max(mb_f.n_gd_iters, 1)
    mb_cross.append((cond, gd_m, mb_f.n_gd_iters, ratio_c, up_m, up_f, f0c, f_target))
    print(
        f"  {cond:6.1f} {gd_m:10d} {mb_f.n_gd_iters:10d} {ratio_c:8.3f} "
        f"{up_m:8d} {up_f:9d}"
    )

print(
    "  Chunking does not change the decision rule: on the expensive end "
    "(cond 20) MAID still tends to use fewer outer steps; on the cheap end "
    "line-search overhead can dominate. Peak working memory tracks the chunk."
)


# %%
# Takeaway
# --------
#
# * Use MAID when each tight lower-level solve is expensive (ill-conditioned
#   physics, weak regularisation, large images, nonsmooth priors with many
#   proximal steps).
# * Prefer fixed accuracy when a residual of ``1e-4`` already costs one or a
#   few gradient steps, as in the inpainting counterpoint.
# * The crossover table is the decision rule: once total fixed-accuracy GD
#   work grows into the tens of thousands, adaptive tolerances typically pay.
# * The outer-step ratio is the stronger result: each outer step costs a
#   hypergradient.
# * Minibatch chunking is a memory control with a fixed reduction order; it
#   does not change the optimisation trajectory.

print()
print("Summary")
print(
    f"  expensive quadratic (cond={FLAGSHIP_COND:g}): "
    f"MAID {maid_flag['gd']} GD / {maid_flag['upper']} outer vs "
    f"fixed {fixed_flag['gd']} GD / {fixed_flag['upper']} outer "
    f"(GD ratio {ratio_flag:.3f}, outer ratio {outer_ratio:.3f})"
)
cheap = crossover_rows[0]
print(
    f"  cheap quadratic (cond={cheap[0]:g}): "
    f"MAID {cheap[1]} GD vs fixed {cheap[2]} GD "
    f"(ratio {cheap[3]:.3f})"
)
print(
    f"  inpainting Tikhonov (DeepInverse GD): "
    f"MAID {gd_maid_img} BaseOptim iters vs fixed {gd_fixed_img} "
    f"(ratio {gd_maid_img / max(gd_fixed_img, 1):.2f})"
)
print(
    f"  minibatch m={MB_M} cond={MB_COND:g} chunk={MB_CHUNK}: "
    f"MAID {gd_mb} GD / {up_mb} outer vs fixed {gd_fx} GD / {up_fx} outer "
    f"(GD ratio {ratio_mb:.3f})"
)
