"""
Bilevel learning of a convex ridge regulariser with MAID
========================================================

MAID is the **outer** solver for the bilevel learning problem. It does not
reconstruct images. Reconstruction is residual-stopped DeepInverse FISTA
on the lower level

    h(x, theta) = 1/2 ||A x - y||^2 + R_theta(x) + (gamma/2) ||x||^2

using the regulariser parameters that learning produced (or a grid-tuned
scalar baseline). Reconstruction panels are labelled by the **prior**, not
by the learning algorithm. MAID appears in the learning-cost figure.

Regulariser
-----------
Multiconv convex ridge regulariser matching the reference CRR/WCRR
(LearnedRegularizers ``priors/wcrr.py``): multi-layer convolution, Lipschitz
normalisation, log scaling, smooth L1, zero-mean first layer,
weak_convexity=0. See :mod:`deepinv.optim.bilevel.convex_ridge`.

Problems (colour Set3C crops), primary first
--------------------------------------------
* Gaussian denoising (A = I). Primary problem: mu_data = 1 from the data
  term, so gamma = 0 (no ridge floor competing with the learned prior).
* random-mask inpainting (gamma = 1e-2 floor)
* Gaussian deblurring (gamma = 1e-2 floor)

Train on two images; evaluate PSNR on a held-out third. Every reported PSNR
comes from a final lower-level solve at a tight residual. Residuals are
printed with the distance bound ``||grad h|| / mu``.

Comparisons
-----------
1. Grid-tuned isotropic TV (mandatory baseline): two fixed difference
   filters and one scalar lambda, tuned on the train set. A learned
   convex regulariser that does not beat TV has not earned its parameters.
2. Grid-tuned scalar Tikhonov (second baseline, not a method under test).
3. Fixed residual/hypergradient accuracy with the **same Armijo line
   search** as MAID (isolates adaptive accuracy).
4. Accelerated MAID (adapts eps/delta and step size).
5. Ridge ablation (inpainting, deblur): data term plus the gamma floor
   only, prior off. If the floor alone already explains most of the
   deblurring PSNR, that comparison is labelled uninformative about priors.

Lower-level solver
------------------
Default lower-level solver is FISTA with O'Donoghue-Candes adaptive
restart. A solver audit runs first: adaptive power iteration for
``L_data``, numerical check of ``L_prior``, certificate against a tight
reference solve, residual histories, then GD / FISTA / FISTA_RESTART /
truncated Newton timings. The certificate
``||x - xstar|| <= ||grad h|| / mu`` depends only on the residual.

Both learning arms use the same outer budget ``N_OUTER`` (printed with
the summary table). Absolute PSNR after few outer steps is a short-run
snapshot, not a converged learning claim.

Regulariser init matches the reference CRR/WCRR: ``sigma = 0.1``,
``beta = 4``. Softening those for runtime is not allowed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import deepinv as dinv
from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    ConvexRidgeConfig,
    CRRSampleProblem,
    TikhonovWeightProblem,
    accelerated_maid_config,
    build_crr_minibatch_oracle,
    exp_scaling,
    grid_tune_tv,
    pack_init_theta,
    recon_tv,
    unpack_theta,
)
from deepinv.utils.demo import load_dataset

torch.manual_seed(0)
dtype = torch.float64
device = "cpu"

try:
    ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # sphinx-gallery execs the example rather than importing it, so __file__
    # is undefined during the docs build.
    ROOT = Path.cwd()
FIG_DIR = Path(os.environ.get("FIG_DIR", ROOT / ".scratch" / "figs"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 32x32 colour crops: model fidelity first (reference sigma/beta), then cost.
# Larger crops are fine if the machine allows; do not soften the regulariser.
IMG_SIZE = 32
NOISE = 0.05
KEEP = 0.5
# Colour multiconv matching the reference stack (in_channels=3). Last width
# 32 rather than 64 keeps the CPU example practical; structure is otherwise
# identical to the reference (3 layers, Lip norm, log scaling).
# Reference WCRR init: sigma=0.1 ("we set it always to 0.1"), beta=4.
_CRR_COMMON = dict(
    nb_channels=(3, 4, 8, 32),
    filter_sizes=(5, 5, 5),
    weak_convexity=0.0,
    lip_fft_size=96,
    beta_init=4.0,
    sigma_init=0.1,
)
# Denoising: A = I gives mu_data = 1, so the ridge floor is unnecessary.
CRR_CFG_DENOISE = ConvexRidgeConfig(gamma=0.0, **_CRR_COMMON)
# Inpainting / deblur: data term alone is not strongly convex; keep floor.
CRR_CFG_RIDGE = ConvexRidgeConfig(gamma=1e-2, **_CRR_COMMON)
# Outer budget for a multi-layer prior; stated in every summary table.
# With truncated Newton lower solves, thirty outer steps on 32x32 colour
# are practical on CPU. Absolute PSNR is still a short-run snapshot.
N_OUTER = 30
FIXED_EPS = 2e-5  # per-element rms gradient tolerance
LEARN_EPS0 = 2e-5  # per-element rms gradient tolerance
REPORT_EPS = 2e-6  # per-element rms gradient tolerance
ALPHA0_CRR = 1e-2
# Reference filter scale. Initialising much smaller leaves the prior nearly
# inert: the data term then dominates, the lower level is easy at any
# accuracy, and the adaptive schedule has nothing to adapt to. At 0.02 this
# example reported MAID and fixed accuracy as indistinguishable on denoising
# and MAID as the worse of the two on the ill-conditioned problem, which is
# an artefact of the initialisation rather than a property of either method.
WEIGHT_SCALE = 1.0
# Shared line-search parameters for MAID and the fixed-accuracy arm.
LS_RHO = 0.5
LS_RHO_BAR = 1.2
LS_LAMBD = 0.01
LS_MAX_BT = 12
MAX_GD = 25_000
# Cap for audit GD/vanilla FISTA only (they fail on ill-conditioned paths).
AUDIT_MAX_ITER = 5_000
# Default after the audit: truncated Newton (HVP already used for hypergrad).
LOWER_SOLVER = "NEWTON"


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(torch.mean((a - b) ** 2).item())
    if mse <= 0:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def to_np_img(img: torch.Tensor):
    x = img.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    arr = x.numpy()
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr.clip(0.0, 1.0)


val_transform = transforms.Compose(
    [transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()]
)
dataset = load_dataset("set3c", transform=val_transform)
assert len(dataset) >= 3
train_imgs = [dataset[i].unsqueeze(0).to(device=device, dtype=dtype) for i in (0, 1)]
heldout_img = dataset[2].unsqueeze(0).to(device=device, dtype=dtype)
assert train_imgs[0].shape[1] == 3
print(
    f"train images: {len(train_imgs)} x {tuple(train_imgs[0].shape)}; "
    f"held-out: {tuple(heldout_img.shape)}"
)
print(
    f"CRR multiconv channels={CRR_CFG_RIDGE.nb_channels}, "
    f"filters={CRR_CFG_RIDGE.filter_sizes}, n_params={CRR_CFG_RIDGE.n_params}, "
    f"weak_convexity={CRR_CFG_RIDGE.weak_convexity}, "
    f"sigma_init={CRR_CFG_RIDGE.sigma_init}, beta_init={CRR_CFG_RIDGE.beta_init}"
)
print(
    f"Noise: Gaussian sigma={NOISE}. Denoising uses gamma=0 (mu_data=1 from "
    f"A=I); inpainting/deblur use gamma={CRR_CFG_RIDGE.gamma}."
)
print(
    f"Reported PSNR: final lower-level solve at eps={REPORT_EPS} "
    "(stop when ||grad h|| <= eps * mu; print residual and "
    "||grad h||/mu distance bound)."
)
print(
    f"Lower-level solver: {LOWER_SOLVER} "
    "(certificate is residual-based and solver-independent)."
)


def make_denoise(x: torch.Tensor, seed: int):
    physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(0.0))
    gen = torch.Generator(device=device).manual_seed(seed)
    y = x + NOISE * torch.randn(x.shape, generator=gen, dtype=dtype, device=device)
    return physics, y


def make_inpaint(x: torch.Tensor, seed: int):
    gen = torch.Generator(device=device).manual_seed(seed)
    mask = (
        torch.rand(x.shape, generator=gen, dtype=dtype, device=device) > (1 - KEEP)
    ).to(dtype)
    physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=mask, device=device)
    y = physics(x) + NOISE * torch.randn(
        x.shape, generator=gen, dtype=dtype, device=device
    )
    return physics, y


def make_deblur(x: torch.Tensor, seed: int):
    k = 9
    ax = torch.arange(k, dtype=dtype, device=device) - k // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    filt = torch.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
    filt = (filt / filt.sum()).view(1, 1, k, k)
    physics = dinv.physics.Blur(filter=filt, padding="circular", device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    y = physics(x) + NOISE * torch.randn(
        x.shape, generator=gen, dtype=dtype, device=device
    )
    return physics, y


def build_samples(make_phys, imgs, seed0: int):
    return [(*make_phys(x, seed0 + i), x) for i, x in enumerate(imgs)]


def fmt_res(res: float, mu: float) -> str:
    """Residual with the strong-convexity distance bound."""
    bound = res / max(mu, 1e-30)
    return f"res={res:.2e} dist_bound={bound:.2e} (mu={mu:.3g})"


def make_problem(make_phys, cfg: ConvexRidgeConfig, seed: int = 100):
    x = train_imgs[0]
    physics, y = make_phys(x, seed)
    return physics, y, x


def solver_audit(problem_makers):
    """Prove L_data, L_prior, the certificate, residual history, then solvers.

    Previous softened-regulariser timings are withdrawn: they used
    sigma_init=0.5, beta_init=1.0 and a fixed twelve-iteration L_data
    estimate. Numbers below use the reference chart and the adaptive step.
    """
    print()
    print("=" * 72)
    print("SOLVER AUDIT (reference sigma=0.1, beta=4)")
    print("=" * 72)

    th0_den = pack_init_theta(
        CRR_CFG_DENOISE, dtype=dtype, seed=0, weight_scale=WEIGHT_SCALE
    )
    th0_ridge = pack_init_theta(
        CRR_CFG_RIDGE, dtype=dtype, seed=0, weight_scale=WEIGHT_SCALE
    )

    # ---- 1. Adaptive L_data vs 200 fixed iterations --------------------
    print()
    print("--- 1. L_data: adaptive (+safety) vs 200 fixed power iters ---")
    print(
        f"{'physics':<12} {'L_adapt':>12} {'L_raw':>12} {'n_it':>6} "
        f"{'L_200':>12} {'adapt>=200':>10}"
    )
    L_data_rows = []
    for label, make_phys, cfg in problem_makers:
        physics, y, x = make_problem(make_phys, cfg)
        prob = CRRSampleProblem(physics=physics, y=y, x_star=x, cfg=cfg, max_iter=100)
        L_adapt, n_it, L_raw = prob.estimate_data_lipschitz()
        L_200, _, _ = prob.estimate_data_lipschitz(fixed_iters=200)
        ok = L_adapt + 1e-12 >= L_200
        print(
            f"{label:<12} {L_adapt:12.6f} {L_raw:12.6f} {n_it:6d} "
            f"{L_200:12.6f} {str(ok):>10}"
        )
        L_data_rows.append(
            {
                "label": label,
                "L_adapt": L_adapt,
                "L_raw": L_raw,
                "n_it": n_it,
                "L_200": L_200,
                "ok": ok,
            }
        )
        if not ok:
            raise RuntimeError(
                f"Adaptive L_data {L_adapt} below L_200 {L_200} on {label}"
            )

    # ---- 2. L_prior analytic vs measured Hess R ------------------------
    print()
    print(
        "--- 2. L_prior: analytic 2*exp(beta)*lip(W)/lip_0+gamma "
        "vs measured Hess R ---"
    )
    print(
        f"{'physics':<12} {'analytic':>12} {'measured':>12} "
        f"{'ratio a/m':>10} {'safe':>6}"
    )
    L_prior_rows = []
    for label, make_phys, cfg in problem_makers:
        physics, y, x = make_problem(make_phys, cfg)
        th0 = th0_den if cfg.gamma == 0.0 else th0_ridge
        prob = CRRSampleProblem(physics=physics, y=y, x_star=x, cfg=cfg, max_iter=100)
        prob.load_theta(th0)
        analytic = prob.analytic_prior_lipschitz()
        # zeros: quadratic region maximises smooth_l1 curvature
        measured, n_it, raw = prob.measure_prior_lipschitz(th0, x=torch.zeros_like(x))
        ratio = analytic / max(raw, 1e-30)
        safe = analytic + 1e-12 >= raw
        print(
            f"{label:<12} {analytic:12.4f} {raw:12.4f} " f"{ratio:10.3f} {str(safe):>6}"
        )
        L_prior_rows.append(
            {
                "label": label,
                "analytic": analytic,
                "measured": raw,
                "ratio": ratio,
                "safe": safe,
            }
        )
        if not safe:
            raise RuntimeError(
                f"Analytic L_prior {analytic} < measured {raw} on {label}; "
                "step size is unsafe."
            )

    # ---- 3. Certificate vs reference solve -----------------------------
    print()
    print("--- 3. Certificate: ||x_loose - x_ref|| <= ||grad h|| / mu ---")
    print(
        f"{'physics':<12} {'||dx||':>12} {'bound':>12} {'ratio':>10} " f"{'holds':>6}"
    )
    cert_rows = []
    for label, make_phys, cfg in problem_makers:
        physics, y, x = make_problem(make_phys, cfg)
        th0 = th0_den if cfg.gamma == 0.0 else th0_ridge
        prob = CRRSampleProblem(
            physics=physics,
            y=y,
            x_star=x,
            cfg=cfg,
            max_iter=MAX_GD,
            solver=LOWER_SOLVER,
        )
        x_init = physics.A_adjoint(y).detach().clone()
        x_loose, res_loose = prob.solve_lower(
            th0, eps=1e-2, x_init=x_init.clone(), solver=LOWER_SOLVER
        )
        x_ref, _ = prob.solve_lower(
            th0, eps=1e-6, x_init=x_init.clone(), solver=LOWER_SOLVER
        )
        dist = float((x_loose - x_ref).flatten().norm().item())
        bound = res_loose / max(prob.mu(), 1e-30)
        ratio = dist / max(bound, 1e-30)
        holds = dist <= bound * 1.01
        print(
            f"{label:<12} {dist:12.4e} {bound:12.4e} {ratio:10.4f} " f"{str(holds):>6}"
        )
        cert_rows.append(
            {
                "label": label,
                "dist": dist,
                "bound": bound,
                "ratio": ratio,
                "holds": holds,
            }
        )
        if not holds:
            raise RuntimeError(
                f"Certificate failed on {label}: dist={dist} bound={bound}"
            )

    # ---- 4. Residual histories (GD monotone objective) -----------------
    print()
    print("--- 4. Residual histories (first / mid / last) ---")
    history_rows = []
    for label, make_phys, cfg in problem_makers:
        physics, y, x = make_problem(make_phys, cfg)
        th0 = th0_den if cfg.gamma == 0.0 else th0_ridge
        x_init = physics.A_adjoint(y).detach().clone()
        eps = 1e-3
        # GD: check objective monotonicity. On ill-conditioned paths it may
        # not hit the residual tol within AUDIT_MAX_ITER; still record.
        prob_gd = CRRSampleProblem(
            physics=physics,
            y=y,
            x_star=x,
            cfg=cfg,
            max_iter=AUDIT_MAX_ITER,
            solver="GD",
        )
        mon = None
        hist_gd = CRRSampleProblem._empty_history()
        try:
            out = prob_gd.solve_lower(
                th0,
                eps=eps,
                x_init=x_init.clone(),
                solver="GD",
                record_every=max(1, AUDIT_MAX_ITER // 50),
            )
            _, res_gd, hist_gd = out
            mon = hist_gd["monotone_objective"]
        except RuntimeError:
            # Run a short fixed-iteration GD manually for monotonicity only.
            step = prob_gd._stepsize()
            prob_gd.load_theta(th0)
            xg = x_init.clone()
            prev = float(prob_gd.energy_h(xg, th0, reload=False).item())
            mon = True
            res_list_tmp = []
            for it in range(min(200, AUDIT_MAX_ITER)):
                g = prob_gd.grad_x_h(xg, th0, reload=False)
                xg = xg - step * g
                obj = float(prob_gd.energy_h(xg, th0, reload=False).item())
                if obj > prev + 1e-12:
                    mon = False
                prev = obj
                if it % 20 == 0:
                    res_list_tmp.append(float(g.flatten().norm().item()))
            hist_gd = {
                "iters": list(range(0, len(res_list_tmp) * 20, 20)),
                "residuals": res_list_tmp,
                "objectives": [],
                "monotone_objective": mon,
                "n_iters": 200,
                "n_restarts": 0,
                "n_cg": 0,
            }
        res_list = hist_gd["residuals"]
        it_list = hist_gd["iters"]
        if len(res_list) >= 3:
            mid = len(res_list) // 2
            triple = (res_list[0], res_list[mid], res_list[-1])
            it_triple = (it_list[0], it_list[mid], it_list[-1])
        elif res_list:
            triple = (res_list[0], res_list[-1], res_list[-1])
            it_triple = (it_list[0], it_list[-1], it_list[-1])
        else:
            triple = (float("nan"),) * 3
            it_triple = (0, 0, 0)
        print(
            f"  {label} GD: iters={hist_gd['n_iters']} "
            f"res@it {it_triple} = {[f'{r:.2e}' for r in triple]} "
            f"monotone_obj={mon}",
            flush=True,
        )
        if mon is False:
            raise RuntimeError(
                f"GD objective not monotone on {label}: L is under-estimated."
            )
        # FISTA_RESTART residual trend
        prob_fr = CRRSampleProblem(
            physics=physics,
            y=y,
            x_star=x,
            cfg=cfg,
            max_iter=MAX_GD,
            solver="FISTA_RESTART",
        )
        out_fr = prob_fr.solve_lower(
            th0,
            eps=eps,
            x_init=x_init.clone(),
            solver="FISTA_RESTART",
            record_every=max(1, 50),
        )
        _, res_fr, hist_fr = out_fr
        res_list = hist_fr["residuals"]
        if len(res_list) >= 2:
            trend_ok = res_list[-1] < res_list[0]
        else:
            trend_ok = True
        print(
            f"  {label} FISTA_RESTART: iters={hist_fr['n_iters']} "
            f"restarts={hist_fr.get('n_restarts', 0)} "
            f"res first={res_list[0]:.2e} last={res_list[-1]:.2e} "
            f"trend_down={trend_ok}",
            flush=True,
        )
        history_rows.append(
            {
                "label": label,
                "gd_iters": hist_gd["n_iters"],
                "gd_monotone": mon,
                "fr_iters": hist_fr["n_iters"],
                "fr_restarts": hist_fr.get("n_restarts", 0),
            }
        )

    # ---- 5. Solver comparison at matched residual ----------------------
    print()
    print("--- 5. Solver comparison (eps=1e-3, same init) ---")
    print(f"{'physics':<12} {'solver':<14} {'iters':>8} {'wall_s':>8} " f"{'res':>10}")
    solver_rows = []
    for label, make_phys, cfg in problem_makers:
        physics, y, x = make_problem(make_phys, cfg)
        th0 = th0_den if cfg.gamma == 0.0 else th0_ridge
        x_init = physics.A_adjoint(y).detach().clone()
        eps = 1e-3
        for name in ("GD", "FISTA", "FISTA_RESTART", "NEWTON"):
            # GD / vanilla FISTA may not converge on ill-conditioned paths
            # with reference beta; cap their budget so the audit finishes.
            max_it = MAX_GD if name in ("FISTA_RESTART", "NEWTON") else AUDIT_MAX_ITER
            prob = CRRSampleProblem(
                physics=physics,
                y=y,
                x_star=x,
                cfg=cfg,
                max_iter=max_it,
                solver=name,
            )
            t0 = time.perf_counter()
            try:
                _, res = prob.solve_lower(
                    th0, eps=eps, x_init=x_init.clone(), solver=name
                )
                wall = time.perf_counter() - t0
                ok = True
                err = ""
            except RuntimeError as exc:
                wall = time.perf_counter() - t0
                res = float("nan")
                ok = False
                err = str(exc)[:80]
            print(
                f"{label:<12} {name:<14} {prob.n_gd_iters:8d} "
                f"{wall:8.2f} {res:10.2e}" + (f"  FAIL {err}" if not ok else ""),
                flush=True,
            )
            solver_rows.append(
                {
                    "label": label,
                    "solver": name,
                    "iters": prob.n_gd_iters,
                    "wall": wall,
                    "res": res,
                    "ok": ok,
                }
            )
    print(
        f"Default lower-level solver for learning: {LOWER_SOLVER}",
        flush=True,
    )
    print("=" * 72)
    return {
        "L_data": L_data_rows,
        "L_prior": L_prior_rows,
        "certificate": cert_rows,
        "history": history_rows,
        "solvers": solver_rows,
    }


def grid_tune_tikhonov(samples, n_grid: int = 9):
    log_lams = torch.linspace(-4.0, 1.0, n_grid, dtype=dtype)
    best = None
    for log_lam in log_lams:
        th = log_lam.clone()
        losses, psnrs = [], []
        for physics, y, x_star in samples:
            prob = TikhonovWeightProblem(
                physics=physics, y=y, x_star=x_star, solver="GD", max_iter=MAX_GD
            )
            xh, _ = prob.solve_lower(th, eps=REPORT_EPS)
            losses.append(float(prob.g(xh).item()))
            psnrs.append(psnr(xh, x_star))
        mean_f = sum(losses) / len(losses)
        mean_p = sum(psnrs) / len(psnrs)
        if best is None or mean_f < best["f"]:
            best = {
                "log_lam": float(log_lam),
                "lam": float(torch.exp(log_lam)),
                "f": mean_f,
                "psnr_train": mean_p,
            }
    return best


def recon_tikhonov_converged(physics, y, x_star, log_lam: float):
    th = torch.tensor(log_lam, dtype=dtype)
    prob = TikhonovWeightProblem(
        physics=physics, y=y, x_star=x_star, solver="GD", max_iter=MAX_GD
    )
    xh, res = prob.solve_lower(th, eps=REPORT_EPS)
    mu = float(prob.mu(th))
    tol = REPORT_EPS * mu * (xh.numel() ** 0.5)
    if res > tol * 1.01:
        raise RuntimeError(f"Tikhonov recon not converged: {res} > {tol}")
    return xh, psnr(xh, x_star), float(res), float(prob.g(xh).item()), mu


def recon_crr_converged(
    physics,
    y,
    x_star,
    theta: torch.Tensor,
    cfg: ConvexRidgeConfig,
    lip_theta: torch.Tensor | None = None,
):
    """Residual-stopped CRR recon under the frozen Lip chart.

    ``lip_theta`` is the parameter vector at which Lip normalisation is
    frozen (the training initialisation). If omitted, Lip freezes at
    ``theta`` itself. Using a different freeze point than training
    redefines the model.
    """
    prob = CRRSampleProblem(
        physics=physics,
        y=y,
        x_star=x_star,
        cfg=cfg,
        max_iter=MAX_GD,
        solver=LOWER_SOLVER,
    )
    freeze = lip_theta if lip_theta is not None else theta
    prob.load_theta(freeze)
    xh, res = prob.solve_lower(theta, eps=REPORT_EPS)
    mu = float(prob.mu(theta))
    tol = REPORT_EPS * mu * (xh.numel() ** 0.5)
    if res > tol * 1.01:
        raise RuntimeError(f"CRR recon not converged: {res} > {tol}")
    return xh, psnr(xh, x_star), float(res), float(prob.g(xh).item()), mu


def recon_ridge_only(physics, y, x_star, gamma: float):
    """Data term plus gamma floor only (learned prior switched off).

    For gamma > 0 this is Tikhonov with lambda = gamma. For gamma = 0 and
    Denoising (A = I) the least-squares solution is the measurement y, so
    the ridge-only PSNR equals the noisy-input PSNR.
    """
    if gamma <= 0.0:
        xh = y.detach().clone()
        # Residual of pure data fidelity at x = y for A = I is zero.
        mu = 1.0
        res = 0.0
        return xh, psnr(xh, x_star), res, 0.0, mu
    log_lam = float(torch.log(torch.tensor(gamma, dtype=dtype)).item())
    return recon_tikhonov_converged(physics, y, x_star, log_lam)


def _mean_upper_level(oracle, theta: torch.Tensor, eps: float) -> float:
    """Mean g(x_hat) over samples after a residual-stopped lower solve."""
    vals = []
    for o in oracle.sample_oracles:
        xv, _ = o.problem.solve_lower(theta, eps=eps)
        vals.append(float(o.problem.g(xv).item()))
    return sum(vals) / len(vals)


def train_crr_maid(
    samples, cfg: ConvexRidgeConfig, accelerated: bool, n_outer: int = N_OUTER
):
    oracle = build_crr_minibatch_oracle(
        samples, cfg=cfg, chunk_size=1, max_iter=MAX_GD, solver=LOWER_SOLVER
    )
    th0 = pack_init_theta(cfg, dtype=dtype, seed=0, weight_scale=WEIGHT_SCALE)
    # Freeze Lip once at theta_0 on every sample problem. Later outer
    # steps keep that constant so the hypergradient matches the solve.
    for o in oracle.sample_oracles:
        o.problem.load_theta(th0)
    common = dict(
        eps0=LEARN_EPS0,
        delta0=LEARN_EPS0,
        alpha0=ALPHA0_CRR,
        rho=LS_RHO,
        rho_bar=LS_RHO_BAR,
        nu=0.5,
        nu_bar=1.05,
        eta=0.5,
        lambd=LS_LAMBD,
        max_BT=LS_MAX_BT,
        max_iter=n_outer,
        tol=1e-5,
        g_convex=True,
        check_descent_direction=False,
        eps_min=1e-5,
        delta_min=1e-5,
        max_outer_BT=25,
        nonmonotone=False,
    )
    cfg = accelerated_maid_config(**common) if accelerated else MAIDConfig(**common)
    t0 = time.perf_counter()
    th, hist = MAID(oracle, cfg).run(th0.clone())
    wall = time.perf_counter() - t0
    gd = sum(int(getattr(o.problem, "n_gd_iters", 0)) for o in oracle.sample_oracles)
    return {
        "theta": th.detach(),
        "theta0": th0.detach(),
        "gd": gd,
        "wall": wall,
        "bt": int(hist["n_backtrack_failures"]),
        "n_outer": int(hist["n_upper_iters"]),
        "f_trace": list(hist["f_exact"]),
    }


def train_crr_fixed(samples, cfg: ConvexRidgeConfig, n_outer: int = N_OUTER):
    """Fixed residual and hypergradient accuracy; same Armijo line search.

    Isolates adaptive accuracy (the MAID contribution). Both arms share
    alpha0, rho, rho_bar, lambd and max_BT. This arm holds eps = delta =
    FIXED_EPS every outer step and only backtracks the step size. The
    previous plain loop ``th = th - alpha0 * z`` confounded accuracy
    adaptation with step-size adaptation.
    """
    oracle = build_crr_minibatch_oracle(
        samples, cfg=cfg, chunk_size=1, max_iter=MAX_GD, solver=LOWER_SOLVER
    )
    th = pack_init_theta(cfg, dtype=dtype, seed=0, weight_scale=WEIGHT_SCALE)
    th0 = th.detach().clone()
    for o in oracle.sample_oracles:
        o.problem.load_theta(th0)
    warm = None
    f_trace = []
    n_bt = 0
    alpha = ALPHA0_CRR
    L_g = 1.0  # upper level g = 0.5 ||x - x_star||^2
    t0 = time.perf_counter()
    for _ in range(n_outer):
        lower = oracle.solve_lower_level(th, eps=FIXED_EPS, warm_start=warm)
        hyper = oracle.hypergradient(th, lower, delta=FIXED_EPS)
        z = hyper.z
        z_norm_sq = float(torch.sum(z * z).item())
        g_old = float(oracle.g(lower.x).item())
        gg_old = float(oracle.grad_g(lower.x).norm().item())
        # Same sandwich as MAID (g_convex path) at constant accuracy.
        U_lower = g_old - gg_old * FIXED_EPS - 0.5 * L_g * (FIXED_EPS**2)
        alpha_try = alpha
        line_ok = False
        lower_trial = lower
        for _bt in range(LS_MAX_BT):
            th_trial = th - alpha_try * z
            lower_trial = oracle.solve_lower_level(
                th_trial, eps=FIXED_EPS, warm_start=lower
            )
            g_new = float(oracle.g(lower_trial.x).item())
            gg_new = float(oracle.grad_g(lower_trial.x).norm().item())
            # Fixed accuracy: no nu_bar inflation of the trial residual.
            U_upper = g_new + gg_new * FIXED_EPS + 0.5 * L_g * (FIXED_EPS**2)
            psi = U_upper - U_lower + LS_LAMBD * alpha_try * z_norm_sq
            psi = psi - 0.5 * L_g * (FIXED_EPS**2)
            if psi <= 0.0:
                line_ok = True
                break
            alpha_try *= LS_RHO
            n_bt += 1
        if line_ok:
            th = th - alpha_try * z
            warm = lower_trial
            alpha = LS_RHO_BAR * alpha_try
        else:
            # Line search exhausted; keep theta, shrink starting alpha.
            n_bt += 1
            alpha = max(ALPHA0_CRR * 1e-3, alpha * LS_RHO)
        f_trace.append(_mean_upper_level(oracle, th, eps=5e-3))
    wall = time.perf_counter() - t0
    gd = sum(int(getattr(o.problem, "n_gd_iters", 0)) for o in oracle.sample_oracles)
    return {
        "theta": th.detach(),
        "theta0": th0,
        "gd": gd,
        "wall": wall,
        "bt": n_bt,
        "n_outer": n_outer,
        "f_trace": f_trace,
    }


def eval_set_converged(
    samples,
    theta_crr,
    log_lam_tik,
    cfg: ConvexRidgeConfig,
    lip_theta: torch.Tensor | None = None,
):
    rows = []
    for physics, y, x_star in samples:
        xt, pt, rt, ft, mt = recon_tikhonov_converged(physics, y, x_star, log_lam_tik)
        xc, pc, rc, fc, mc = recon_crr_converged(
            physics, y, x_star, theta_crr, cfg, lip_theta=lip_theta
        )
        psnr_noisy = psnr(y, x_star)
        rows.append(
            {
                "x": x_star,
                "y": y,
                "tik": xt,
                "crr": xc,
                "psnr_tik": pt,
                "psnr_crr": pc,
                "psnr_noisy": psnr_noisy,
                "res_tik": rt,
                "res_crr": rc,
                "mu_tik": mt,
                "mu_crr": mc,
                "bound_tik": rt / max(mt, 1e-30),
                "bound_crr": rc / max(mc, 1e-30),
                "f_tik": ft,
                "f_crr": fc,
            }
        )
    return rows


def mean_key(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def print_scaling_table(label, th0, th1, cfg: ConvexRidgeConfig):
    s0 = exp_scaling(th0, cfg).detach().cpu()
    s1 = exp_scaling(th1, cfg).detach().cpu()
    ratios = s1 / s0
    print(f"  {label} exp(scaling) motion (acceptance test):", flush=True)
    print(
        f"  {'k':>4} {'exp(s0)':>12} {'exp(s_final)':>14} {'ratio':>10}",
        flush=True,
    )
    n_show = min(8, cfg.n_filters)
    for k in range(n_show):
        print(
            f"  {k:4d} {float(s0[k]):12.6f} {float(s1[k]):14.6f} "
            f"{float(ratios[k]):10.4f}",
            flush=True,
        )
    if cfg.n_filters > n_show:
        print(
            f"  ... ({cfg.n_filters - n_show} further filters omitted)",
            flush=True,
        )
    print(
        f"  ratio range [{float(ratios.min()):.4f}, {float(ratios.max()):.4f}], "
        f"mean exp(s_final)={float(s1.mean()):.6f}, std={float(s1.std()):.6f}",
        flush=True,
    )
    # Knee probe on the first training image at final theta: fraction of
    # filter responses with |exp(beta) * exp(s) * (W x)| >= 1.
    from deepinv.optim.bilevel.convex_ridge import apply_conv, get_conv_lip

    w, s_t, b = unpack_theta(th1, cfg)
    lip = get_conv_lip(w, cfg, detach=True)
    feats = apply_conv(train_imgs[0], w, cfg, lip=lip)
    inner = torch.exp(b) * feats * torch.exp(s_t)
    frac_lin = float((inner.abs() >= 1.0).float().mean().item())
    print(
        f"  knee probe on train[0]: frac |exp(beta)*exp(s)*Wx| >= 1 "
        f"= {frac_lin:.3f} (beta={float(b):.3f})",
        flush=True,
    )
    if float(ratios.max()) < 1.05 and float(ratios.min()) > 0.95:
        if frac_lin < 0.05:
            print(
                "  Scale flat and responses are quadratic: s cancels in "
                "smooth_l1 there (see convex_ridge module docstring).",
                flush=True,
            )
        else:
            print(
                "  Scale nearly flat despite responses past the knee "
                f"(frac={frac_lin:.3f}). Identifiable in principle; joint "
                "Euclidean steps on flat theta under-move s relative to "
                "the weight block (||z_w|| dominates ||z_s||).",
                flush=True,
            )
    else:
        print(
            "  Scaling moved: responses reached the smooth_l1 knee and s "
            "is identifiable at the reference sigma/beta.",
            flush=True,
        )
    return s0, s1, ratios


# Denoising first (primary). gamma=0 because A=I gives mu_data=1.
problem_specs = [
    ("denoising", make_denoise, 50, CRR_CFG_DENOISE),
    ("inpainting", make_inpaint, 100, CRR_CFG_RIDGE),
    ("deblur", make_deblur, 200, CRR_CFG_RIDGE),
]

summary = []
artifacts = {}
t_demo0 = time.perf_counter()
# Full solver audit before any learning numbers.
audit = solver_audit(
    [
        ("denoising", make_denoise, CRR_CFG_DENOISE),
        ("inpainting", make_inpaint, CRR_CFG_RIDGE),
        ("deblur", make_deblur, CRR_CFG_RIDGE),
    ]
)
solver_ablation_rows = [r for r in audit["solvers"] if r["label"] == "inpainting"]

for pname, make_phys, seed0, crr_cfg in problem_specs:
    print()
    print(
        f"=== {pname} (gamma={crr_cfg.gamma}, n_params={crr_cfg.n_params}) ===",
        flush=True,
    )
    train_samples = build_samples(make_phys, train_imgs, seed0)
    held_samples = build_samples(make_phys, [heldout_img], seed0 + 50)

    # Noisy / measurement PSNR on held-out (for denoising this is the gain
    # baseline; for other problems it is the raw observation quality).
    held_y_psnr = psnr(held_samples[0][1], held_samples[0][2])
    print(
        f"  held-out measurement PSNR={held_y_psnr:.2f} dB " f"(noise sigma={NOISE})",
        flush=True,
    )

    print("Grid-tuning isotropic TV baseline on train set ...", flush=True)
    t0 = time.perf_counter()
    tv = grid_tune_tv(train_samples, n_grid=15, verify_once=True)
    print(
        f"  TV lambda={tv['lam']:.4f} train f={tv['f']:.2f} "
        f"train PSNR={tv['psnr_train']:.2f} "
        f"wall={time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    # Held-out TV recon at the train-tuned lambda.
    tv_held_rows = []
    for physics, y, x_star in held_samples:
        x_tv, p_tv, info_tv = recon_tv(physics, y, x_star, tv["lam"])
        tv_held_rows.append({"tv": x_tv, "psnr_tv": p_tv, "info_tv": info_tv})
    tv_psnr_held = mean_key(tv_held_rows, "psnr_tv")
    print(
        f"  TV held PSNR={tv_psnr_held:.2f} (lam={tv['lam']:.4f})",
        flush=True,
    )

    print("Grid-tuning scalar Tikhonov baseline on train set ...", flush=True)
    t0 = time.perf_counter()
    tik = grid_tune_tikhonov(train_samples, n_grid=9)
    print(
        f"  Tikhonov lambda={tik['lam']:.4f} (log={tik['log_lam']:.3f}) "
        f"train f={tik['f']:.2f} train PSNR={tik['psnr_train']:.2f} "
        f"wall={time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    # Ridge ablation: prior off, data term + gamma floor only.
    ridge_rows = []
    for physics, y, x_star in held_samples:
        xr, pr, rr, fr, mr = recon_ridge_only(physics, y, x_star, crr_cfg.gamma)
        ridge_rows.append(
            {
                "ridge": xr,
                "psnr_ridge": pr,
                "res_ridge": rr,
                "mu_ridge": mr,
                "bound_ridge": rr / max(mr, 1e-30),
            }
        )
    ridge_psnr_held = mean_key(ridge_rows, "psnr_ridge")
    ridge_res_held = mean_key(ridge_rows, "res_ridge")
    ridge_mu_held = mean_key(ridge_rows, "mu_ridge")
    print(
        f"  ridge ablation (prior off, gamma={crr_cfg.gamma}): "
        f"held PSNR={ridge_psnr_held:.2f} "
        f"({fmt_res(ridge_res_held, ridge_mu_held)})",
        flush=True,
    )

    print(
        f"Bilevel learning of CRR with accelerated MAID " f"(N_OUTER={N_OUTER}) ...",
        flush=True,
    )
    maid = train_crr_maid(train_samples, crr_cfg, accelerated=True, n_outer=N_OUTER)
    print(
        f"  MAID (learning) outer={maid['n_outer']} GD={maid['gd']} "
        f"BT={maid['bt']} wall={maid['wall']:.1f}s",
        flush=True,
    )
    s0, s1, ratios = print_scaling_table("MAID", maid["theta0"], maid["theta"], crr_cfg)
    maid["s0"], maid["s1"], maid["ratios"] = s0, s1, ratios

    print(
        f"Bilevel learning of CRR with fixed eps/delta={FIXED_EPS} "
        f"and the same Armijo line search (N_OUTER={N_OUTER}) ...",
        flush=True,
    )
    fixed = train_crr_fixed(train_samples, crr_cfg, n_outer=N_OUTER)
    print(
        f"  fixed+LS outer={fixed['n_outer']} GD={fixed['gd']} "
        f"BT={fixed['bt']} wall={fixed['wall']:.1f}s",
        flush=True,
    )

    print(f"Final recon at eps={REPORT_EPS} ...", flush=True)
    # Freeze Lip at the training initialisation so the report evaluates
    # the same model the hypergradient differentiated.
    train_eval_maid = eval_set_converged(
        train_samples,
        maid["theta"],
        tik["log_lam"],
        crr_cfg,
        lip_theta=maid["theta0"],
    )
    held_eval_maid = eval_set_converged(
        held_samples,
        maid["theta"],
        tik["log_lam"],
        crr_cfg,
        lip_theta=maid["theta0"],
    )
    train_eval_fixed = eval_set_converged(
        train_samples,
        fixed["theta"],
        tik["log_lam"],
        crr_cfg,
        lip_theta=fixed["theta0"],
    )
    held_eval_fixed = eval_set_converged(
        held_samples,
        fixed["theta"],
        tik["log_lam"],
        crr_cfg,
        lip_theta=fixed["theta0"],
    )

    row = {
        "problem": pname,
        "gamma": crr_cfg.gamma,
        "y_psnr_held": held_y_psnr,
        "tv_lam": tv["lam"],
        "tv_psnr_train": tv["psnr_train"],
        "tv_psnr_held": tv_psnr_held,
        "tik_lam": tik["lam"],
        "tik_psnr_train": mean_key(train_eval_maid, "psnr_tik"),
        "tik_psnr_held": mean_key(held_eval_maid, "psnr_tik"),
        "tik_res_held": mean_key(held_eval_maid, "res_tik"),
        "tik_bound_held": mean_key(held_eval_maid, "bound_tik"),
        "tik_mu_held": mean_key(held_eval_maid, "mu_tik"),
        "maid_psnr_train": mean_key(train_eval_maid, "psnr_crr"),
        "maid_psnr_held": mean_key(held_eval_maid, "psnr_crr"),
        "maid_res_held": mean_key(held_eval_maid, "res_crr"),
        "maid_bound_held": mean_key(held_eval_maid, "bound_crr"),
        "maid_mu_held": mean_key(held_eval_maid, "mu_crr"),
        "fixed_psnr_train": mean_key(train_eval_fixed, "psnr_crr"),
        "fixed_psnr_held": mean_key(held_eval_fixed, "psnr_crr"),
        "fixed_res_held": mean_key(held_eval_fixed, "res_crr"),
        "fixed_bound_held": mean_key(held_eval_fixed, "bound_crr"),
        "fixed_mu_held": mean_key(held_eval_fixed, "mu_crr"),
        "ridge_psnr_held": ridge_psnr_held,
        "ridge_res_held": ridge_res_held,
        "ridge_bound_held": mean_key(ridge_rows, "bound_ridge"),
        "ridge_mu_held": ridge_mu_held,
        "maid_gd": maid["gd"],
        "fixed_gd": fixed["gd"],
        "maid_wall": maid["wall"],
        "fixed_wall": fixed["wall"],
        "maid_bt": maid["bt"],
    }
    # If ridge alone already gets most of the reconstruction PSNR, the
    # prior comparison is uninformative on that problem.
    best_prior = max(row["tik_psnr_held"], row["maid_psnr_held"], row["tv_psnr_held"])
    if best_prior > 1e-8:
        ridge_frac = row["ridge_psnr_held"] / best_prior
    else:
        ridge_frac = 0.0
    row["ridge_frac_of_best"] = ridge_frac
    row["ridge_uninformative"] = pname in ("deblur", "inpainting") and ridge_frac >= 0.9
    summary.append(row)
    artifacts[pname] = {
        "cfg": crr_cfg,
        "tv": tv,
        "tv_held_rows": tv_held_rows,
        "tik": tik,
        "maid": maid,
        "fixed": fixed,
        "ridge_rows": ridge_rows,
        "train_eval_maid": train_eval_maid,
        "held_eval_maid": held_eval_maid,
        "held_eval_fixed": held_eval_fixed,
    }
    print(
        f"  TRAIN  TV={row['tv_psnr_train']:.2f}  "
        f"Tikhonov={row['tik_psnr_train']:.2f}  "
        f"CRR MAID={row['maid_psnr_train']:.2f}  "
        f"CRR fixed={row['fixed_psnr_train']:.2f}",
        flush=True,
    )
    print(
        f"  HELD   y={row['y_psnr_held']:.2f}  "
        f"TV={row['tv_psnr_held']:.2f}  "
        f"Tikhonov={row['tik_psnr_held']:.2f} "
        f"({fmt_res(row['tik_res_held'], row['tik_mu_held'])})  "
        f"CRR MAID={row['maid_psnr_held']:.2f} "
        f"({fmt_res(row['maid_res_held'], row['maid_mu_held'])})  "
        f"CRR fixed={row['fixed_psnr_held']:.2f} "
        f"({fmt_res(row['fixed_res_held'], row['fixed_mu_held'])})  "
        f"ridge-only={row['ridge_psnr_held']:.2f}",
        flush=True,
    )
    if row["ridge_uninformative"]:
        print(
            f"  NOTE: ridge floor alone reaches "
            f"{100.0 * ridge_frac:.0f}% of the best held PSNR on {pname}; "
            "prior comparison is uninformative on this problem.",
            flush=True,
        )


CAPTION_RECON = (
    f"Reconstruction by residual-stopped {LOWER_SOLVER} (eps={REPORT_EPS}); "
    "panels differ only in the regulariser parameters. "
    "dist_bound = ||grad h|| / mu. "
    f"sigma_init={CRR_CFG_RIDGE.sigma_init}, beta_init={CRR_CFG_RIDGE.beta_init}, "
    f"N_OUTER={N_OUTER}, train_n={len(train_imgs)}, img={IMG_SIZE}."
)


def show_img(ax, img, title):
    arr = to_np_img(img)
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(arr, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def write_heldout_figure(pname: str, title: str, out_name: str):
    he = artifacts[pname]["held_eval_maid"][0]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, (img, panel) in zip(
        axes,
        [
            (he["x"], "ground truth"),
            (
                he["y"],
                f"measurement\nPSNR {he['psnr_noisy']:.2f} dB",
            ),
            (
                he["tik"],
                f"grid-tuned scalar prior\nPSNR {he['psnr_tik']:.2f} dB\n"
                f"res {he['res_tik']:.1e}\n"
                f"dist {he['bound_tik']:.1e}",
            ),
            (
                he["crr"],
                f"learned convex ridge prior\nPSNR {he['psnr_crr']:.2f} dB\n"
                f"res {he['res_crr']:.1e}\n"
                f"dist {he['bound_crr']:.1e}",
            ),
        ],
        strict=True,
    ):
        show_img(ax, img, panel)
    fig.suptitle(
        f"{title}: same solver, different regulariser parameters\n" + CAPTION_RECON,
        fontsize=10,
    )
    fig.tight_layout()
    path = FIG_DIR / out_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


write_heldout_figure(
    "denoising",
    "Denoising held-out (gamma=0, primary)",
    "maid_crr_denoising_heldout.png",
)
write_heldout_figure(
    "inpainting",
    "Inpainting held-out",
    "maid_crr_inpainting_heldout.png",
)
write_heldout_figure(
    "deblur",
    "Deblur held-out",
    "maid_crr_deblur_heldout.png",
)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, pname in zip(axes, ("denoising", "inpainting", "deblur"), strict=True):
    maid = artifacts[pname]["maid"]
    fixed = artifacts[pname]["fixed"]
    if maid["f_trace"]:
        ax.plot(
            range(1, len(maid["f_trace"]) + 1),
            maid["f_trace"],
            "-o",
            label="accelerated MAID",
            markersize=4,
        )
    if fixed["f_trace"]:
        ax.plot(
            range(1, len(fixed["f_trace"]) + 1),
            fixed["f_trace"],
            "-s",
            label="fixed eps/delta + line search",
            markersize=4,
        )
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("upper-level cost f")
    ax.set_title(f"{pname}: learning efficiency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle(
    f"Learning efficiency (N_OUTER={N_OUTER}): MAID adapts eps/delta; "
    f"fixed arm holds eps=delta={FIXED_EPS}; both use Armijo line search",
    fontsize=10,
)
fig.tight_layout()
path = FIG_DIR / "maid_crr_learning_cost.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}", flush=True)

# Filters from the primary problem (denoising).
th = artifacts["denoising"]["maid"]["theta"]
cfg_den = artifacts["denoising"]["cfg"]
weights, scaling, beta = unpack_theta(th, cfg_den)
w0 = weights[0].detach().cpu()
n_out = min(w0.shape[0], 4)
fig, axes = plt.subplots(n_out, 3, figsize=(6, 2 * n_out))
for i in range(n_out):
    for c in range(3):
        ax = axes[i, c] if n_out > 1 else axes[c]
        k = w0[i, c].numpy()
        vmax = max(abs(k.min()), abs(k.max()), 1e-8)
        ax.imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(
            f"out={i} ch={c} exp(s)={float(torch.exp(scaling[0, i])):.3f}",
            fontsize=7,
        )
        ax.axis("off")
fig.suptitle(
    "Learned CRR first-layer filters (denoising; parameters from accelerated MAID)",
    fontsize=10,
)
fig.tight_layout()
path = FIG_DIR / "maid_crr_filters_denoising.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}", flush=True)

print()
print(
    f"Summary (sigma_init={CRR_CFG_RIDGE.sigma_init}, "
    f"beta_init={CRR_CFG_RIDGE.beta_init}, N_OUTER={N_OUTER}, "
    f"train_n={len(train_imgs)}, img={IMG_SIZE}x{IMG_SIZE}, "
    f"lower_solver={LOWER_SOLVER}; matched Armijo line search; "
    "only eps/delta adaptivity differs)"
)
print(
    f"{'problem':<12} {'prior / learning':<36} {'PSNR_train':>10} "
    f"{'PSNR_held':>10} {'res_held':>10} {'dist_bnd':>10} "
    f"{'GD':>8} {'wall_s':>8}"
)
for row in summary:
    print(
        f"{row['problem']:<12} {'measurement (noisy / y)':<36} "
        f"{'-':>10} {row['y_psnr_held']:10.2f} "
        f"{'-':>10} {'-':>10} {'-':>8} {'-':>8}"
    )
    print(
        f"{row['problem']:<12} {'ridge floor only (prior off)':<36} "
        f"{'-':>10} {row['ridge_psnr_held']:10.2f} "
        f"{row['ridge_res_held']:10.2e} {row['ridge_bound_held']:10.2e} "
        f"{'-':>8} {'-':>8}"
    )
    print(
        f"{row['problem']:<12} {'grid-tuned isotropic TV':<36} "
        f"{row['tv_psnr_train']:10.2f} {row['tv_psnr_held']:10.2f} "
        f"{'-':>10} {'-':>10} "
        f"{'-':>8} {'-':>8}"
    )
    print(
        f"{row['problem']:<12} {'grid-tuned scalar Tikhonov':<36} "
        f"{row['tik_psnr_train']:10.2f} {row['tik_psnr_held']:10.2f} "
        f"{row['tik_res_held']:10.2e} {row['tik_bound_held']:10.2e} "
        f"{'-':>8} {'-':>8}"
    )
    print(
        f"{row['problem']:<12} {'CRR fixed eps/delta + line search':<36} "
        f"{row['fixed_psnr_train']:10.2f} {row['fixed_psnr_held']:10.2f} "
        f"{row['fixed_res_held']:10.2e} {row['fixed_bound_held']:10.2e} "
        f"{row['fixed_gd']:8d} {row['fixed_wall']:8.1f}"
    )
    print(
        f"{row['problem']:<12} {'CRR accelerated MAID':<36} "
        f"{row['maid_psnr_train']:10.2f} {row['maid_psnr_held']:10.2f} "
        f"{row['maid_res_held']:10.2e} {row['maid_bound_held']:10.2e} "
        f"{row['maid_gd']:8d} {row['maid_wall']:8.1f}"
    )

demo_wall = time.perf_counter() - t_demo0
print()
print("Notes")
print(
    f"  Train set: {len(train_imgs)} Set3C colour {IMG_SIZE}x{IMG_SIZE} images; "
    "held-out: 1 image."
)
print(
    f"  Noise: Gaussian sigma={NOISE}. Denoising is primary: gamma=0 because "
    "A=I gives mu_data=1 from the data term; the ridge floor is not needed."
)
print(
    f"  Outer budget: N_OUTER={N_OUTER} for both learning arms "
    f"(small for a multi-layer prior; do not over-read absolute PSNR)."
)
print(
    "  Comparison design (option 1): both arms use the same Armijo line "
    f"search (alpha0={ALPHA0_CRR}, rho={LS_RHO}, lambd={LS_LAMBD}, "
    f"max_BT={LS_MAX_BT}). Fixed arm holds eps=delta={FIXED_EPS}; "
    "MAID adapts residual and hypergradient accuracy. Isolates adaptive "
    "accuracy from step-size adaptation."
)
print(
    "  TV baseline: isotropic total variation, lambda grid-tuned on train; "
    "mandatory comparison for a learned multi-filter prior."
)
print(
    "  Scalar Tikhonov baseline: log-spaced grid over lambda; second "
    "baseline, not a method under test."
)
print(
    "  Lip normalisation is frozen at theta_0 for the outer run "
    "(forward solve, HVP and mixed Jacobian share one constant). "
    "L_prior uses 2*exp(beta)*lip(W)/lip_0+gamma for the step size."
)
print(
    f"  CRR: multiconv {CRR_CFG_RIDGE.nb_channels}, Lip-normalised, log scaling, "
    f"smooth L1, weak_convexity={CRR_CFG_RIDGE.weak_convexity}, "
    f"sigma_init={CRR_CFG_RIDGE.sigma_init}, beta_init={CRR_CFG_RIDGE.beta_init}, "
    f"n_params={CRR_CFG_RIDGE.n_params}; gamma=0 on denoising, "
    f"gamma={CRR_CFG_RIDGE.gamma} on inpainting/deblur."
)
print(
    f"  Reported PSNR: final lower-level solve at eps={REPORT_EPS}; residual "
    "is ||grad h|| and must meet eps*mu. Distance bound ||grad h||/mu is "
    "printed beside every residual (the interpretable certificate)."
)
print(
    f"  Lower-level solver: {LOWER_SOLVER}. "
    "Certificate ||x-xstar|| <= ||grad h||/mu is residual-based and "
    "independent of the algorithm that produced x."
)
print(
    "  Previous softened-regulariser results (sigma_init=0.5, beta_init=1.0) "
    "and fixed twelve-iteration L_data estimates are withdrawn."
)
if solver_ablation_rows:
    parts = []
    for name in ("GD", "FISTA", "FISTA_RESTART", "NEWTON"):
        r = next((x for x in solver_ablation_rows if x["solver"] == name), None)
        if r is not None and r.get("ok", True):
            parts.append(f"{name} {r['iters']} iters / {r['wall']:.2f}s")
    if parts:
        print("  Solver ablation (one inpaint, eps=1e-3): " + "; ".join(parts) + ".")
for row in summary:
    if row["ridge_uninformative"]:
        print(
            f"  Ridge ablation on {row['problem']}: floor alone reaches "
            f"{100.0 * row['ridge_frac_of_best']:.0f}% of best held PSNR "
            f"({row['ridge_psnr_held']:.2f} of "
            f"{max(row['tik_psnr_held'], row['maid_psnr_held']):.2f}); "
            "prior comparison is uninformative on this problem."
        )
    else:
        print(
            f"  Ridge ablation on {row['problem']}: floor-only held PSNR "
            f"{row['ridge_psnr_held']:.2f} dB "
            f"({100.0 * row['ridge_frac_of_best']:.0f}% of best prior)."
        )
print(
    "  MAID appears in the learning-cost figure and the wall/GD columns; "
    "reconstruction panels are labelled by the prior, not by MAID."
)
print(
    "  Log-scale s: cancels exactly in the quadratic region of smooth_l1; "
    "identifiability depends on sigma relative to filter response scale "
    "(see convex_ridge module docstring)."
)
print(f"  Example wall-clock: {demo_wall:.1f}s (~{demo_wall/60:.1f} min).")
print(f"  Figures: {FIG_DIR}")

if "denoising" in artifacts:
    ratios = artifacts["denoising"]["maid"]["ratios"]
    s1 = artifacts["denoising"]["maid"]["s1"]
    print()
    print("Denoising CRR scaling motion (exp(s)) at reference sigma/beta:")
    print(
        f"  min ratio={float(ratios.min()):.4f}, max ratio={float(ratios.max()):.4f}, "
        f"std(exp(s_final))={float(s1.std()):.6f}"
    )
    spread = float(s1.max() / s1.min())
    print(
        f"  final exp(s) spread max/min={spread:.3f} "
        f"({'similar across filters' if spread < 1.2 else 'allocated across filters'})"
    )
    if float(ratios.max()) < 1.05 and float(ratios.min()) > 0.95:
        print(
            "  Scale flat at this operating point (quadratic region of "
            "smooth_l1 relative to current filter responses)."
        )
    else:
        print(
            "  Scale moved: knee reached at reference sigma=0.1; s is " "identifiable."
        )
