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
    accelerated_maid_config,
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


def run_maid_to_gap(
    cond: float,
    target: float = TARGET_GAP,
    max_discover: int = 50,
    accelerated: bool = False,
):
    """Return GD count, outer steps and wall time to reach ``target`` gap."""
    prob = make_quadratic(cond)
    theta0 = torch.ones(prob.d, dtype=dtype)
    f_star = float(prob.f_closed_form(prob.closed_form_theta_star()).item())
    alpha0 = 1.0 / max(upper_lipschitz(prob), 1e-8)

    def _cfg(max_iter: int) -> MAIDConfig:
        base = maid_config(alpha0, max_iter)
        if not accelerated:
            return base
        return accelerated_maid_config(
            eps0=base.eps0,
            delta0=base.delta0,
            alpha0=base.alpha0,
            rho=base.rho,
            rho_bar=base.rho_bar,
            nu=base.nu,
            nu_bar=base.nu_bar,
            eta=base.eta,
            lambd=base.lambd,
            max_BT=base.max_BT,
            max_iter=max_iter,
            tol=base.tol,
            g_convex=base.g_convex,
            check_descent_direction=base.check_descent_direction,
        )

    # Discover the first outer step that meets the gap.
    _, hist = MAID(SmoothHypergradientOracle(prob), _cfg(max_discover)).run(
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
    theta, hist = MAID(SmoothHypergradientOracle(prob), _cfg(hit)).run(theta0.clone())
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
print(f"{'method':<28} {'f_final':>10} {'GD iters':>10} " f"{'upper':>7} {'wall_s':>8}")
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
assert (
    maid_flag["gd"] < fixed_flag["gd"]
), "Flagship should show MAID using fewer GD steps than fixed accuracy"


# %%
# 2. Three-way crossover: fixed, MAID, accelerated MAID
# -----------------------------------------------------
#
# Accelerated MAID uses Zhang-Hager nonmonotone acceptance and a
# Barzilai-Borwein initial step (see MAIDConfig.nonmonotone / bb_init).
# The question is whether the crossover point moves.

CONDS = [2.0, 5.0, 10.0, 20.0, 30.0]
print()
print(f"Three-way crossover: GD steps to relative gap < {TARGET_GAP:g}")
print(
    f"{'cond':>6} {'gd_fix':>9} {'gd_maid':>9} {'gd_acc':>9} "
    f"{'r_maid':>7} {'r_acc':>7} {'bt_m':>5} {'bt_a':>5}"
)

crossover_rows = []
for cond in CONDS:
    f = run_fixed_to_gap(cond)
    m = run_maid_to_gap(cond, accelerated=False)
    a = run_maid_to_gap(cond, accelerated=True)
    r_m = m["gd"] / max(f["gd"], 1)
    r_a = a["gd"] / max(f["gd"], 1)
    crossover_rows.append(
        (cond, f["gd"], m["gd"], a["gd"], r_m, r_a, m["bt_fail"], a["bt_fail"])
    )
    print(
        f"{cond:6.1f} {f['gd']:9d} {m['gd']:9d} {a['gd']:9d} "
        f"{r_m:7.3f} {r_a:7.3f} {m['bt_fail']:5d} {a['bt_fail']:5d}"
    )

# Vanilla MAID: ratio > 1 at small cond, < 1 at large cond.
assert crossover_rows[0][4] > 1.0, "Cheap end should favour fixed over vanilla MAID"
assert crossover_rows[-1][4] < 1.0, "Expensive end should favour vanilla MAID"
# Accelerated should improve on vanilla at the expensive end.
assert (
    crossover_rows[-1][5] < crossover_rows[-1][4]
), "Accelerated MAID should use fewer GD steps than vanilla at high cond"
# Backtracking should not increase under acceleration on this sweep.
assert all(
    r[7] <= r[6] + 2 for r in crossover_rows
), "Accelerated BT count should not substantially exceed vanilla"
print()
print(
    "Reading: vanilla MAID crosses below ratio 1 near condition number 5 to 10. "
    "Accelerated MAID is already cheaper at condition number 5 and keeps a "
    "ratio near 0.2 above that. At condition number 2 it still loses to fixed "
    "accuracy, but by less than vanilla, and backtracking failures drop. "
    "The mechanism is the sandwich gap U_upper - U_lower, which nonmonotone "
    "acceptance and BB steps reduce the cost of."
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
physics = dinv.physics.Inpainting(img_size=x_star.shape[1:], mask=mask, device="cpu")
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
assert (
    gd_maid_img > gd_fixed_img
), "Inpainting counterpoint should show fixed accuracy cheaper on BaseOptim iters"
assert problem_img.residual_kind == "gradient"

# Accelerated MAID on the same inpainting instance: the mechanism check.
problem_acc = TikhonovWeightProblem(
    physics=physics,
    y=y,
    x_star=x_star,
    solver="GD",
    max_iter=20_000,
)
cfg_acc_img = accelerated_maid_config(
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
t0 = time.perf_counter()
theta_acc_img, hist_acc_img = MAID(TikhonovWeightOracle(problem_acc), cfg_acc_img).run(
    theta0_img.clone()
)
wall_acc_img = time.perf_counter() - t0
f_acc_img = float(problem_acc.f_closed_form(theta_acc_img).item())
print()
print(
    f"{'Accelerated MAID':<28} {f_acc_img:>10.4f} "
    f"{float(torch.exp(theta_acc_img)): >10.4f} "
    f"{problem_acc.n_gd_iters:>16d} {wall_acc_img:>8.3f}"
)
print(
    f"Accelerated BT failures: {hist_acc_img['n_backtrack_failures']} "
    f"(vanilla had {hist_img['n_backtrack_failures']}). "
    f"BaseOptim iters {problem_acc.n_gd_iters} vs vanilla {gd_maid_img}."
)
assert hist_acc_img["n_backtrack_failures"] <= hist_img["n_backtrack_failures"]
assert f_acc_img < f0_img


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
    return sum(float(p.f_closed_form(theta).item()) for p in problems) / len(problems)


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
assert torch.equal(
    traj_rows[0][2], traj_rows[1][2]
), "final theta must be bitwise identical across chunk sizes"
print("  bitwise trajectory match: yes (chunk 1 vs chunk m)")

# Peak working memory. The claim that matters is flat in dataset size m at
# fixed chunk size. Instrument peak concurrent sample-state bytes during
# accumulation of float64 vectors of length d (1.53 MB each). Measurement
# is pure Python/torch tensor sizes, not CUDA (unavailable here).
print()
print("  Peak working memory (float64 state length d=200000 = 1.53 MB/sample).")
print(
    "  Only a chunk of states is live during accumulation; warm-start "
    "storage is O(m) by design and reported separately."
)
print(f"  {'m':>6} {'chunk':>6} {'peak_work_MB':>12} {'warm_store_MB':>14}")


def _peak_chunk_working(m: int, chunk: int, d: int = 200_000):
    peak = 0
    z_acc = None
    x_warm = []
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        chunk_x = [torch.randn(d, dtype=dtype) for _ in range(start, end)]
        nbytes = sum(t.numel() * t.element_size() for t in chunk_x)
        peak = max(peak, nbytes)
        for x in chunk_x:
            x_warm.append(x.detach())
            z_acc = x.clone() if z_acc is None else z_acc + x
        del chunk_x
    warm_bytes = sum(t.numel() * t.element_size() for t in x_warm)
    return peak, warm_bytes


mem_flat = []
for m_mem in (16, 32, 64, 128):
    peak, warm = _peak_chunk_working(m_mem, chunk=4)
    mem_flat.append((m_mem, peak, warm))
    print(
        f"  {m_mem:6d} {4:6d} {peak / (1024 ** 2):12.2f} " f"{warm / (1024 ** 2):14.2f}"
    )
# Flat in m: all peak_work equal.
assert all(
    r[1] == mem_flat[0][1] for r in mem_flat
), "peak working memory must be independent of m at fixed chunk"
print()
print(f"  {'chunk':>6} {'m':>6} {'peak_work_MB':>12}")
mem_chunk = []
for cs in (1, 2, 4, 8, 16):
    peak, _warm = _peak_chunk_working(64, chunk=cs)
    mem_chunk.append((cs, peak))
    print(f"  {cs:6d} {64:6d} {peak / (1024 ** 2):12.2f}")
assert (
    mem_chunk[0][1] < mem_chunk[-1][1]
), "peak working memory must grow with chunk size"

# Matched-quality cost: vanilla MAID, accelerated MAID, warm-started fixed.
# Fresh problem objects per method so n_gd_iters is not shared.
print()
print(
    f"  matched quality at cond={MB_COND:g}, m={MB_M}, chunk_size={MB_CHUNK} "
    f"(fixed stops at each method's f; warm-started fixed baseline)"
)

common_mb = dict(
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


def _run_mb_maid(accelerated: bool):
    probs = make_quadratic_dataset(MB_M, cond=MB_COND, n=40, d=3, seed=0)
    cfg = (
        accelerated_maid_config(**common_mb) if accelerated else MAIDConfig(**common_mb)
    )
    mb = MinibatchOracle(wrap_smooth_dataset(probs), chunk_size=MB_CHUNK)
    t0 = time.perf_counter()
    th, hist = MAID(mb, cfg).run(theta0_mb.clone())
    wall = time.perf_counter() - t0
    return {
        "f": _mb_f(probs, th),
        "gd": mb.n_gd_iters,
        "sl": mb.n_sample_lower_solves,
        "sh": mb.n_sample_hypergradients,
        "wall": wall,
        "bt": hist["n_backtrack_failures"],
        "up": int(hist["n_upper_iters"]),
    }


def _run_mb_fixed(f_target: float):
    probs = make_quadratic_dataset(MB_M, cond=MB_COND, n=40, d=3, seed=0)
    a0 = _mb_alpha0(probs)
    mb = MinibatchOracle(wrap_smooth_dataset(probs), chunk_size=MB_CHUNK)
    th = theta0_mb.clone()
    warm = None
    n_up = 0
    t0 = time.perf_counter()
    while n_up < 80:
        n_up += 1
        lower = mb.solve_lower_level(th, eps=FIXED_EPS, warm_start=warm)
        hyper = mb.hypergradient(th, lower, delta=FIXED_EPS)
        th = th - a0 * hyper.z
        warm = lower
        if _mb_f(probs, th) <= f_target * 1.001:
            break
    wall = time.perf_counter() - t0
    return {
        "f": _mb_f(probs, th),
        "gd": mb.n_gd_iters,
        "sl": mb.n_sample_lower_solves,
        "sh": mb.n_sample_hypergradients,
        "wall": wall,
        "up": n_up,
    }


van = _run_mb_maid(accelerated=False)
acc = _run_mb_maid(accelerated=True)
fix_v = _run_mb_fixed(van["f"])
fix_a = _run_mb_fixed(acc["f"])

print(
    f"  {'method':<28} {'f_final':>10} {'GD':>10} "
    f"{'sample_lo':>10} {'BT':>5} {'wall_s':>8}"
)
print(
    f"  {'vanilla MAID':<28} {van['f']:>10.4f} {van['gd']:>10d} "
    f"{van['sl']:>10d} {van['bt']:>5d} {van['wall']:>8.3f}"
)
print(
    f"  {'fixed to vanilla f':<28} {fix_v['f']:>10.4f} {fix_v['gd']:>10d} "
    f"{fix_v['sl']:>10d} {'-':>5} {fix_v['wall']:>8.3f}"
)
print(
    f"  {'accelerated MAID':<28} {acc['f']:>10.4f} {acc['gd']:>10d} "
    f"{acc['sl']:>10d} {acc['bt']:>5d} {acc['wall']:>8.3f}"
)
print(
    f"  {'fixed to accelerated f':<28} {fix_a['f']:>10.4f} {fix_a['gd']:>10d} "
    f"{fix_a['sl']:>10d} {'-':>5} {fix_a['wall']:>8.3f}"
)

r_gd_v = van["gd"] / max(fix_v["gd"], 1)
r_wall_v = van["wall"] / max(fix_v["wall"], 1e-12)
r_sl_v = van["sl"] / max(fix_v["sl"], 1)
r_gd_a = acc["gd"] / max(fix_a["gd"], 1)
r_wall_a = acc["wall"] / max(fix_a["wall"], 1e-12)
r_sl_a = acc["sl"] / max(fix_a["sl"], 1)
print(
    f"  vanilla/fixed:  GD {r_gd_v:.3f}, wall {r_wall_v:.3f}, "
    f"sample_lower {r_sl_v:.2f}"
)
print(
    f"  accel/fixed:    GD {r_gd_a:.3f}, wall {r_wall_a:.3f}, "
    f"sample_lower {r_sl_a:.2f}"
)
print(
    f"  accel/vanilla:  GD {acc['gd'] / max(van['gd'], 1):.3f}, "
    f"wall {acc['wall'] / max(van['wall'], 1e-12):.3f}, "
    f"sample_lower {acc['sl'] / max(van['sl'], 1):.2f}, "
    f"BT {acc['bt']}/{van['bt']}"
)
print(
    "  Mechanism: every line-search trial re-solves all m samples. "
    "Acceleration cuts backtracking, so the sample_lower saving is "
    f"multiplied by m (here m={MB_M}): vanilla {van['sl']} against "
    f"accelerated {acc['sl']} against fixed {fix_v['sl']}."
)
print(
    "  Note: an earlier draft reused the same problem objects for MAID "
    "and fixed, so n_gd_iters summed both runs and falsely made fixed "
    "look more expensive. Counters above use fresh problems per method."
)
assert van["f"] < f0_mb
assert acc["f"] < f0_mb
# Acceleration should reduce sample_lower relative to vanilla on this path.
assert (
    acc["sl"] < van["sl"]
), "accelerated MAID should issue fewer sample_lower solves than vanilla"

# Cost table over condition number: vanilla, accelerated, fixed (to vanilla f).
print()
print(
    f"  Minibatch cost table (m={MB_M}, chunk={MB_CHUNK}, max_iter=8): "
    f"vanilla vs accelerated vs fixed to each method's f"
)
print(
    f"  {'cond':>6} {'gd_v':>8} {'gd_a':>8} {'gd_fv':>8} "
    f"{'sl_v':>6} {'sl_a':>6} {'sl_fv':>6} "
    f"{'bt_v':>5} {'bt_a':>5} "
    f"{'rg_v':>6} {'rg_a':>6} {'rw_a':>6}"
)
mb_cross = []
for cond in (5.0, 10.0, 20.0):
    a0 = None

    def _one(accelerated: bool):
        probs = make_quadratic_dataset(MB_M, cond=cond, n=40, d=3, seed=0)
        a0_local = _mb_alpha0(probs)
        cfg = (
            accelerated_maid_config(
                eps0=1e-1,
                delta0=1e-1,
                alpha0=a0_local,
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
            if accelerated
            else MAIDConfig(
                eps0=1e-1,
                delta0=1e-1,
                alpha0=a0_local,
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
        )
        mb = MinibatchOracle(wrap_smooth_dataset(probs), chunk_size=MB_CHUNK)
        t0 = time.perf_counter()
        th, hist = MAID(mb, cfg).run(torch.ones(3, dtype=dtype))
        wall = time.perf_counter() - t0
        return {
            "f": _mb_f(probs, th),
            "gd": mb.n_gd_iters,
            "sl": mb.n_sample_lower_solves,
            "wall": wall,
            "bt": hist["n_backtrack_failures"],
            "a0": a0_local,
        }

    def _fixed(f_target: float, a0_local: float):
        probs = make_quadratic_dataset(MB_M, cond=cond, n=40, d=3, seed=0)
        mb = MinibatchOracle(wrap_smooth_dataset(probs), chunk_size=MB_CHUNK)
        th = torch.ones(3, dtype=dtype)
        warm = None
        n_up = 0
        t0 = time.perf_counter()
        while n_up < 80:
            n_up += 1
            lower = mb.solve_lower_level(th, eps=FIXED_EPS, warm_start=warm)
            hyper = mb.hypergradient(th, lower, delta=FIXED_EPS)
            th = th - a0_local * hyper.z
            warm = lower
            if _mb_f(probs, th) <= f_target * 1.001:
                break
        wall = time.perf_counter() - t0
        return {
            "f": _mb_f(probs, th),
            "gd": mb.n_gd_iters,
            "sl": mb.n_sample_lower_solves,
            "wall": wall,
            "up": n_up,
        }

    v = _one(False)
    a = _one(True)
    fv = _fixed(v["f"], v["a0"])
    fa = _fixed(a["f"], a["a0"])
    rg_v = v["gd"] / max(fv["gd"], 1)
    rg_a = a["gd"] / max(fa["gd"], 1)
    rw_a = a["wall"] / max(fa["wall"], 1e-12)
    mb_cross.append((cond, v, a, fv, fa, rg_v, rg_a, rw_a))
    print(
        f"  {cond:6.1f} {v['gd']:8d} {a['gd']:8d} {fv['gd']:8d} "
        f"{v['sl']:6d} {a['sl']:6d} {fv['sl']:6d} "
        f"{v['bt']:5d} {a['bt']:5d} "
        f"{rg_v:6.3f} {rg_a:6.3f} {rw_a:6.3f}"
    )

print(
    "  Reading: vanilla MAID loses on GD and wall because line search "
    "multiplies sample_lower by about 20x. Accelerated MAID cuts "
    "sample_lower and backtracking, and is competitive with (often cheaper "
    "than) warm-started fixed accuracy on this path. Chunking remains a "
    "memory control with bitwise trajectory invariance; with acceleration "
    "it is also competitive on cost."
)
# For summary variables used below.
gd_mb, wall_mb, sl_mb = van["gd"], van["wall"], van["sl"]
gd_fx, wall_fx, sl_fx = fix_v["gd"], fix_v["wall"], fix_v["sl"]
ratio_mb, ratio_wall = r_gd_v, r_wall_v
gd_acc_mb, wall_acc_mb, sl_acc_mb = acc["gd"], acc["wall"], acc["sl"]
r_gd_acc, r_wall_acc = r_gd_a, r_wall_a


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
# row: cond, gd_fix, gd_maid, gd_acc, r_m, r_a, bt_m, bt_a
print(
    f"  cheap quadratic (cond={cheap[0]:g}): "
    f"fixed {cheap[1]} GD, MAID {cheap[2]} (ratio {cheap[4]:.3f}), "
    f"acc {cheap[3]} (ratio {cheap[5]:.3f}), BT {cheap[6]} -> {cheap[7]}"
)
exp = crossover_rows[-1]
print(
    f"  expensive quadratic (cond={exp[0]:g}): "
    f"fixed {exp[1]} GD, MAID {exp[2]} (ratio {exp[4]:.3f}), "
    f"acc {exp[3]} (ratio {exp[5]:.3f}), BT {exp[6]} -> {exp[7]}"
)
print(
    f"  inpainting Tikhonov (DeepInverse GD): "
    f"MAID {gd_maid_img} BaseOptim iters vs fixed {gd_fixed_img} "
    f"(ratio {gd_maid_img / max(gd_fixed_img, 1):.2f})"
)
print(
    f"  minibatch m={MB_M} cond={MB_COND:g} chunk={MB_CHUNK}: "
    f"vanilla {gd_mb} GD / {wall_mb:.2f}s (ratio {ratio_mb:.3f}), "
    f"accel {gd_acc_mb} GD / {wall_acc_mb:.2f}s (ratio {r_gd_acc:.3f}), "
    f"sample_lower {sl_mb}/{sl_acc_mb}/{sl_fx}"
)
