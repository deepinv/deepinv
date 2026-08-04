"""
Learned convex ridge regulariser trained by MAID
================================================

Trains a Goujon-Unser convex ridge regulariser

    R_theta(x) = sum_k lambda_k sum_j rho((W_k * x)_j)
    rho(t)     = mu log cosh(t / mu)
    lambda_k   = exp(vartheta_k)

(8 kernels of 5x5, 208 parameters, plus gamma ||x||^2 / 2 for strong
convexity) by bilevel learning with MAID on two DeepInverse problems:

* random-mask inpainting
* Gaussian deblurring

on Set3C greyscale crops. Training uses two images; evaluation reports
PSNR on a held-out third image the learning never saw.

Comparisons (per problem, grid-tuned scalar baseline is independent of MAID):

1. Best hand-tuned scalar Tikhonov weight (log-spaced grid on the training
   set upper-level loss).
2. Fixed-accuracy bilevel learning of the same CRR.
3. Accelerated MAID learning of the CRR.

Figures in ``.scratch/figs/``:

* Tikhonov inpainting recon (kept: shows spatial-prior limitation)
* CRR inpainting recon (train and held-out)
* CRR deblur recon (held-out)
* learned kernels

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
    pack_init_theta,
    unpack_theta,
)
from deepinv.utils.demo import load_dataset


torch.manual_seed(0)
dtype = torch.float64
device = "cpu"

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(os.environ.get("FIG_DIR", ROOT / ".scratch" / "figs"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 64
NOISE = 0.05
KEEP = 0.5
CRR_CFG = ConvexRidgeConfig(
    n_kernels=8, kernel_size=5, n_channels=1, mu_rho=0.05, gamma=1e-2
)
N_OUTER = 6
FIXED_EPS = 1e-3
ALPHA0_CRR = 5e-4
ALPHA0_TIK = 1e-3
MAX_GD = 2_000


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(torch.mean((a - b) ** 2).item())
    if mse <= 0:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def to_np(img: torch.Tensor):
    return img.detach().cpu().squeeze().numpy()


# %%
# Data: train on images 0 and 1, hold out image 2
# -----------------------------------------------

val_transform = transforms.Compose(
    [
        transforms.CenterCrop(IMG_SIZE),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)
dataset = load_dataset("set3c", transform=val_transform)
assert len(dataset) >= 3, "Set3C must provide at least 3 images"
train_imgs = [
    dataset[i].unsqueeze(0).to(device=device, dtype=dtype) for i in (0, 1)
]
heldout_img = dataset[2].unsqueeze(0).to(device=device, dtype=dtype)
print(
    f"train images: {len(train_imgs)} x {tuple(train_imgs[0].shape)}; "
    f"held-out: {tuple(heldout_img.shape)}"
)
print(
    f"CRR: {CRR_CFG.n_kernels} kernels of {CRR_CFG.kernel_size}x{CRR_CFG.kernel_size}, "
    f"n_params={CRR_CFG.n_params}, gamma={CRR_CFG.gamma}, mu_rho={CRR_CFG.mu_rho}"
)
print("lambda_k = exp(vartheta_k) (same positivity pattern as TikhonovWeightProblem)")


def make_inpaint(x: torch.Tensor, seed: int):
    gen = torch.Generator(device=device).manual_seed(seed)
    mask = (
        torch.rand(x.shape, generator=gen, dtype=dtype, device=device) > (1 - KEEP)
    ).to(dtype)
    physics = dinv.physics.Inpainting(
        img_size=x.shape[1:], mask=mask, device=device
    )
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
    samples = []
    for i, x in enumerate(imgs):
        physics, y = make_phys(x, seed0 + i)
        samples.append((physics, y, x))
    return samples


# %%
# Grid-tuned scalar Tikhonov (per problem, on training set)
# ---------------------------------------------------------


def grid_tune_tikhonov(samples, n_grid: int = 13):
    """Minimise mean train upper-level loss over log-spaced lambda."""
    log_lams = torch.linspace(-4.0, 1.0, n_grid, dtype=dtype)
    best = None
    for log_lam in log_lams:
        th = log_lam.clone()
        losses = []
        psnrs = []
        for physics, y, x_star in samples:
            prob = TikhonovWeightProblem(
                physics=physics, y=y, x_star=x_star, solver="GD", max_iter=MAX_GD
            )
            xh, _ = prob.solve_lower(th, eps=1e-4)
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


def recon_tikhonov(physics, y, x_star, log_lam: float):
    th = torch.tensor(log_lam, dtype=dtype)
    prob = TikhonovWeightProblem(
        physics=physics, y=y, x_star=x_star, solver="GD", max_iter=MAX_GD
    )
    xh, _ = prob.solve_lower(th, eps=1e-4)
    return xh, psnr(xh, x_star), float(prob.g(xh).item())


def recon_crr(physics, y, x_star, theta: torch.Tensor):
    prob = CRRSampleProblem(
        physics=physics,
        y=y,
        x_star=x_star,
        cfg=CRR_CFG,
        max_iter=MAX_GD,
    )
    xh, _ = prob.solve_lower(theta, eps=1e-4)
    return xh, psnr(xh, x_star), float(prob.g(xh).item())


# %%
# Train CRR with accelerated MAID and fixed-accuracy bilevel
# ----------------------------------------------------------


def train_crr_maid(samples, accelerated: bool, n_outer: int = N_OUTER):
    oracle = build_crr_minibatch_oracle(
        samples, cfg=CRR_CFG, chunk_size=1, max_iter=MAX_GD
    )
    th0 = pack_init_theta(CRR_CFG, dtype=dtype, seed=0, log_lambda0=-1.0)
    common = dict(
        eps0=5e-2,
        delta0=5e-2,
        alpha0=ALPHA0_CRR,
        rho=0.5,
        rho_bar=1.2,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=10,
        max_iter=n_outer,
        tol=1e-5,
        g_convex=True,
        check_descent_direction=False,
        eps_min=1e-5,
        delta_min=1e-5,
        max_outer_BT=20,
    )
    cfg = (
        accelerated_maid_config(**common)
        if accelerated
        else MAIDConfig(**common)
    )
    t0 = time.perf_counter()
    th, hist = MAID(oracle, cfg).run(th0.clone())
    wall = time.perf_counter() - t0
    gd = sum(
        int(getattr(o.problem, "n_gd_iters", 0)) for o in oracle.sample_oracles
    )
    return {
        "theta": th.detach(),
        "gd": gd,
        "wall": wall,
        "bt": int(hist["n_backtrack_failures"]),
        "n_outer": int(hist["n_upper_iters"]),
        "f_trace": list(hist["f_exact"]),
    }


def train_crr_fixed(samples, n_outer: int = N_OUTER):
    oracle = build_crr_minibatch_oracle(
        samples, cfg=CRR_CFG, chunk_size=1, max_iter=MAX_GD
    )
    th = pack_init_theta(CRR_CFG, dtype=dtype, seed=0, log_lambda0=-1.0)
    warm = None
    t0 = time.perf_counter()
    for _ in range(n_outer):
        lower = oracle.solve_lower_level(th, eps=FIXED_EPS, warm_start=warm)
        hyper = oracle.hypergradient(th, lower, delta=FIXED_EPS)
        th = th - ALPHA0_CRR * hyper.z
        warm = lower
    wall = time.perf_counter() - t0
    gd = sum(
        int(getattr(o.problem, "n_gd_iters", 0)) for o in oracle.sample_oracles
    )
    return {"theta": th.detach(), "gd": gd, "wall": wall, "bt": 0, "n_outer": n_outer}


def eval_set(samples, theta_crr, log_lam_tik):
    rows = []
    for physics, y, x_star in samples:
        xt, pt, ft = recon_tikhonov(physics, y, x_star, log_lam_tik)
        xc, pc, fc = recon_crr(physics, y, x_star, theta_crr)
        rows.append(
            {
                "x": x_star,
                "y": y,
                "tik": xt,
                "crr": xc,
                "psnr_tik": pt,
                "psnr_crr": pc,
                "f_tik": ft,
                "f_crr": fc,
            }
        )
    return rows


def mean_psnr(rows, key):
    return sum(r[key] for r in rows) / len(rows)


# %%
# Run both problems
# -----------------

problem_specs = [
    ("inpainting", make_inpaint, 100),
    ("deblur", make_deblur, 200),
]

summary = []
artifacts = {}

for pname, make_phys, seed0 in problem_specs:
    print()
    print(f"=== {pname} ===", flush=True)
    train_samples = build_samples(make_phys, train_imgs, seed0)
    held_samples = build_samples(make_phys, [heldout_img], seed0 + 50)

    print("Grid-tuning Tikhonov on train set ...", flush=True)
    t0 = time.perf_counter()
    tik = grid_tune_tikhonov(train_samples, n_grid=11)
    print(
        f"  best lambda={tik['lam']:.4f} (log={tik['log_lam']:.3f}) "
        f"train f={tik['f']:.2f} train PSNR={tik['psnr_train']:.2f} "
        f"wall={time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    print("Training CRR with accelerated MAID ...", flush=True)
    maid = train_crr_maid(train_samples, accelerated=True, n_outer=N_OUTER)
    print(
        f"  MAID outer={maid['n_outer']} GD={maid['gd']} BT={maid['bt']} "
        f"wall={maid['wall']:.1f}s",
        flush=True,
    )

    print("Training CRR with fixed accuracy ...", flush=True)
    fixed = train_crr_fixed(train_samples, n_outer=N_OUTER)
    print(
        f"  fixed outer={fixed['n_outer']} GD={fixed['gd']} "
        f"wall={fixed['wall']:.1f}s",
        flush=True,
    )

    train_eval_maid = eval_set(
        train_samples, maid["theta"], tik["log_lam"]
    )
    held_eval_maid = eval_set(
        held_samples, maid["theta"], tik["log_lam"]
    )
    train_eval_fixed = eval_set(
        train_samples, fixed["theta"], tik["log_lam"]
    )
    held_eval_fixed = eval_set(
        held_samples, fixed["theta"], tik["log_lam"]
    )

    row = {
        "problem": pname,
        "tik_lam": tik["lam"],
        "tik_psnr_train": mean_psnr(train_eval_maid, "psnr_tik"),
        "tik_psnr_held": mean_psnr(held_eval_maid, "psnr_tik"),
        "maid_psnr_train": mean_psnr(train_eval_maid, "psnr_crr"),
        "maid_psnr_held": mean_psnr(held_eval_maid, "psnr_crr"),
        "fixed_psnr_train": mean_psnr(train_eval_fixed, "psnr_crr"),
        "fixed_psnr_held": mean_psnr(held_eval_fixed, "psnr_crr"),
        "maid_gd": maid["gd"],
        "fixed_gd": fixed["gd"],
        "maid_wall": maid["wall"],
        "fixed_wall": fixed["wall"],
        "maid_bt": maid["bt"],
    }
    summary.append(row)
    artifacts[pname] = {
        "tik": tik,
        "maid": maid,
        "fixed": fixed,
        "train_eval_maid": train_eval_maid,
        "held_eval_maid": held_eval_maid,
        "held_eval_fixed": held_eval_fixed,
    }
    print(
        f"  TRAIN  Tikhonov PSNR={row['tik_psnr_train']:.2f}  "
        f"CRR-MAID={row['maid_psnr_train']:.2f}  "
        f"CRR-fixed={row['fixed_psnr_train']:.2f}",
        flush=True,
    )
    print(
        f"  HELD   Tikhonov PSNR={row['tik_psnr_held']:.2f}  "
        f"CRR-MAID={row['maid_psnr_held']:.2f}  "
        f"CRR-fixed={row['fixed_psnr_held']:.2f}",
        flush=True,
    )


# %%
# Figures
# -------

# Keep Tikhonov inpainting figure (scalar prior limitation).
he = artifacts["inpainting"]["held_eval_maid"][0]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
panels = [
    (he["x"], "ground truth"),
    (he["y"], "measurement"),
    (he["tik"], f"grid Tikhonov\nPSNR {he['psnr_tik']:.2f} dB"),
    (he["crr"], f"CRR-MAID\nPSNR {he['psnr_crr']:.2f} dB"),
]
for ax, (img, title) in zip(axes, panels):
    ax.imshow(to_np(img), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.suptitle(
    "Inpainting held-out: grid-tuned Tikhonov vs learned convex ridge regulariser",
    fontsize=11,
)
fig.tight_layout()
path = FIG_DIR / "maid_crr_inpainting_heldout.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}", flush=True)

# Deblur held-out comparison.
he = artifacts["deblur"]["held_eval_maid"][0]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
panels = [
    (he["x"], "ground truth"),
    (he["y"], "measurement"),
    (he["tik"], f"grid Tikhonov\nPSNR {he['psnr_tik']:.2f} dB"),
    (he["crr"], f"CRR-MAID\nPSNR {he['psnr_crr']:.2f} dB"),
]
for ax, (img, title) in zip(axes, panels):
    ax.imshow(to_np(img), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.suptitle(
    "Deblur held-out: grid-tuned Tikhonov vs learned convex ridge regulariser",
    fontsize=11,
)
fig.tight_layout()
path = FIG_DIR / "maid_crr_deblur_heldout.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}", flush=True)

# Side-by-side Tikhonov-only inpainting (scalar limitation) already in
# maid_imaging_inpainting_recon.png; also save a labelled pair comparing
# train Tikhonov static vs CRR.
te = artifacts["inpainting"]["train_eval_maid"][0]
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, (img, title) in zip(
    axes,
    [
        (te["x"], "ground truth (train)"),
        (te["tik"], f"grid Tikhonov\nPSNR {te['psnr_tik']:.2f} dB"),
        (te["crr"], f"CRR-MAID\nPSNR {te['psnr_crr']:.2f} dB"),
    ],
):
    ax.imshow(to_np(img), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.suptitle(
    "Inpainting train image: zeroth-order Tikhonov has no spatial coupling; CRR does",
    fontsize=10,
)
fig.tight_layout()
path = FIG_DIR / "maid_crr_inpainting_train_pair.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}", flush=True)

# Learned kernels from inpainting MAID (the data-adaptive picture).
th = artifacts["inpainting"]["maid"]["theta"]
kernels, lambdas = unpack_theta(th, CRR_CFG)
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for i, ax in enumerate(axes.ravel()):
    k = kernels[i, 0].detach().cpu().numpy()
    vmax = max(abs(k.min()), abs(k.max()), 1e-8)
    ax.imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"k={i} lam={float(lambdas[i]):.3f}", fontsize=8)
    ax.axis("off")
fig.suptitle("Learned CRR kernels (inpainting, accelerated MAID)", fontsize=11)
fig.tight_layout()
path = FIG_DIR / "maid_crr_kernels_inpainting.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {path}", flush=True)


# %%
# Summary table
# -------------

print()
print(
    f"{'problem':<12} {'method':<16} {'PSNR_train':>10} {'PSNR_held':>10} "
    f"{'GD':>8} {'wall_s':>8}"
)
for row in summary:
    print(
        f"{row['problem']:<12} {'grid Tikhonov':<16} "
        f"{row['tik_psnr_train']:10.2f} {row['tik_psnr_held']:10.2f} "
        f"{'-':>8} {'-':>8}"
    )
    print(
        f"{row['problem']:<12} {'CRR fixed':<16} "
        f"{row['fixed_psnr_train']:10.2f} {row['fixed_psnr_held']:10.2f} "
        f"{row['fixed_gd']:8d} {row['fixed_wall']:8.1f}"
    )
    print(
        f"{row['problem']:<12} {'CRR MAID':<16} "
        f"{row['maid_psnr_train']:10.2f} {row['maid_psnr_held']:10.2f} "
        f"{row['maid_gd']:8d} {row['maid_wall']:8.1f}"
    )

print()
print("Notes")
print(
    f"  Train set: {len(train_imgs)} Set3C greyscale {IMG_SIZE}x{IMG_SIZE} images; "
    "held-out: 1 image."
)
print(
    f"  Scalar baseline: log-spaced grid over lambda in [e^-4, e^1], "
    "selected by mean train upper-level loss (not taken from a MAID run)."
)
print(
    f"  CRR gamma={CRR_CFG.gamma} (mu floor for residual stopping); "
    "lambda_k=exp(vartheta_k)."
)
print(f"  Figures: {FIG_DIR}")
print(
    "  Numbers are measured. If CRR does not beat grid-tuned Tikhonov on a "
    "problem, that is reported rather than tuned away."
)
