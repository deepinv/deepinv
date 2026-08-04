"""
Bilevel imaging with MAID: inpainting and deblurring
====================================================

Learn a Tikhonov regularisation weight ``lambda = exp(theta)`` for two
DeepInverse inverse problems on a real greyscale test image (Set3C,
128x128):

1. Random-mask inpainting (keep probability 0.5, Gaussian noise sigma 0.05).
2. Gaussian deblurring (9x9 kernel, sigma 1.5, noise sigma 0.05).

Each problem is solved three ways from the same initial ``theta0``:

* fixed residual ``1e-4`` every outer step
* MAID
* accelerated MAID (Zhang-Hager nonmonotone acceptance + BB step init)

Figures are written to ``.scratch/figs/`` (or ``FIG_DIR`` if set):

* one reconstruction panel figure per problem (ground truth, measurement,
  recon at ``theta0``, recon at the MAID-learned parameter, with PSNR)
* one convergence figure (upper-level objective against cumulative
  lower-level GD iterations for the three methods)

Why Tikhonov rather than TV
---------------------------
``TVPrior`` with ``PGD`` and the proximal residual is the natural DeepInverse
choice, but a bilevel IFT hypergradient for a nonsmooth TV weight is not
wired in this contribution (the certified path uses
``H = A^T A + lambda I`` for Tikhonov). Building that residual/IFT path is
out of scope of the imaging demo. Tikhonov still exercises real physics,
real images, and residual-stopped DeepInverse ``GD``.

Supervised upper level on a single image: ``f(theta) = 0.5 ||xhat(theta) - x||^2``.
One image is enough to learn a scalar weight; the figure is the claim.

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
    TikhonovWeightOracle,
    TikhonovWeightProblem,
    accelerated_maid_config,
)
from deepinv.utils.demo import load_dataset


# %%
# Paths and device
# ----------------

torch.manual_seed(0)
dtype = torch.float64
device = "cpu"

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(os.environ.get("FIG_DIR", ROOT / ".scratch" / "figs"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
NOISE_SIGMA = 0.05
THETA0 = 0.0  # lambda = 1
MAX_OUTER = 10
FIXED_EPS = 1e-4
# Hypergradient scale is O(number of pixels); 1e-3 keeps theta steps O(1).
ALPHA0 = 1e-3


# %%
# Data
# ----

val_transform = transforms.Compose(
    [
        transforms.CenterCrop(IMG_SIZE),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)
dataset = load_dataset("set3c", transform=val_transform)
x_star = dataset[0].unsqueeze(0).to(device=device, dtype=dtype)
print(f"image shape {tuple(x_star.shape)}, range [{float(x_star.min()):.3f}, {float(x_star.max()):.3f}]")


def make_inpainting(x: torch.Tensor):
    gen = torch.Generator(device=device).manual_seed(1)
    mask = (torch.rand(x.shape, generator=gen, dtype=dtype, device=device) > 0.5).to(
        dtype
    )
    physics = dinv.physics.Inpainting(
        img_size=x.shape[1:], mask=mask, device=device
    )
    noise = NOISE_SIGMA * torch.randn(
        x.shape, generator=gen, dtype=dtype, device=device
    )
    y = physics(x) + noise
    return physics, y, "inpainting", "keep_prob=0.5"


def make_deblur(x: torch.Tensor):
    k = 9
    ax = torch.arange(k, dtype=dtype, device=device) - k // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    filt = torch.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
    filt = (filt / filt.sum()).view(1, 1, k, k)
    physics = dinv.physics.Blur(
        filter=filt, padding="circular", device=device
    )
    gen = torch.Generator(device=device).manual_seed(2)
    noise = NOISE_SIGMA * torch.randn(
        x.shape, generator=gen, dtype=dtype, device=device
    )
    y = physics(x) + noise
    return physics, y, "deblur", "gaussian_9x9_sigma=1.5"


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(torch.mean((a - b) ** 2).item())
    if mse <= 0.0:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def to_np(img: torch.Tensor):
    return img.detach().cpu().squeeze().numpy()


# %%
# Runners
# -------


def make_problem(physics, y, x) -> TikhonovWeightProblem:
    return TikhonovWeightProblem(
        physics=physics,
        y=y,
        x_star=x,
        solver="GD",
        max_iter=30_000,
    )


def recon_at(problem: TikhonovWeightProblem, theta: torch.Tensor, eps: float = 1e-5):
    x, _ = problem.solve_lower(theta, eps=eps)
    return x


def run_fixed(
    physics,
    y,
    x,
    theta0: torch.Tensor,
    n_outer: int,
    alpha0: float,
):
    """Fixed residual every outer step; track f and cumulative GD."""
    problem = make_problem(physics, y, x)
    oracle = TikhonovWeightOracle(problem)
    theta = theta0.clone()
    warm = None
    f_trace = []
    gd_trace = []
    t0 = time.perf_counter()
    for _ in range(n_outer):
        lower = oracle.solve_lower_level(theta, eps=FIXED_EPS, warm_start=warm)
        hyper = oracle.hypergradient(theta, lower, delta=FIXED_EPS)
        theta = theta - alpha0 * hyper.z
        theta = torch.clamp(theta, -5.0, 5.0)
        warm = lower
        # High-accuracy upper-level value for the plot (not free, but honest).
        x_acc, _ = problem.solve_lower(theta, eps=1e-5)
        f_trace.append(float(problem.g(x_acc).item()))
        gd_trace.append(int(problem.n_gd_iters))
    wall = time.perf_counter() - t0
    x_init = recon_at(make_problem(physics, y, x), theta0)
    x_final = recon_at(problem, theta)
    return {
        "theta": float(theta.item()),
        "lam": float(torch.exp(theta).item()),
        "f": f_trace[-1] if f_trace else float("nan"),
        "psnr": psnr(x_final, x),
        "psnr0": psnr(x_init, x),
        "gd": int(problem.n_gd_iters),
        "wall": wall,
        "f_trace": f_trace,
        "gd_trace": gd_trace,
        "x0": x_init,
        "xhat": x_final,
        "n_outer": n_outer,
        "bt": 0,
    }


def run_maid(
    physics,
    y,
    x,
    theta0: torch.Tensor,
    n_outer: int,
    alpha0: float,
    accelerated: bool,
):
    """MAID / accelerated MAID in one run; record (f, GD) after each outer step."""
    problem = make_problem(physics, y, x)
    oracle = TikhonovWeightOracle(problem)
    common = dict(
        eps0=1e-2,
        delta0=1e-2,
        alpha0=alpha0,
        rho=0.5,
        rho_bar=1.2,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=12,
        tol=1e-6,
        g_convex=True,
        check_descent_direction=False,
        eps_min=1e-6,
        delta_min=1e-6,
        max_outer_BT=30,
    )
    cfg = (
        accelerated_maid_config(max_iter=n_outer, **common)
        if accelerated
        else MAIDConfig(max_iter=n_outer, **common)
    )
    maid = MAID(oracle, cfg)
    gd_marks: list[int] = []
    # _f_diag is called once per accepted outer step; stamp cumulative GD there.
    _orig_f_diag = maid._f_diag

    def _f_diag_mark(theta):
        val = _orig_f_diag(theta)
        gd_marks.append(int(problem.n_gd_iters))
        return val

    maid._f_diag = _f_diag_mark  # type: ignore[method-assign]

    t0 = time.perf_counter()
    theta, hist = maid.run(theta0.clone())
    wall = time.perf_counter() - t0
    theta = torch.clamp(theta, -5.0, 5.0)

    f_trace = list(hist["f_exact"])
    n = min(len(f_trace), len(gd_marks))
    f_trace = f_trace[:n]
    gd_trace = gd_marks[:n]

    x_init = recon_at(make_problem(physics, y, x), theta0)
    x_final = recon_at(problem, theta)
    return {
        "theta": float(theta.item()),
        "lam": float(torch.exp(theta).item()),
        "f": float(problem.g(x_final).item()),
        "psnr": psnr(x_final, x),
        "psnr0": psnr(x_init, x),
        "gd": int(problem.n_gd_iters),
        "wall": wall,
        "bt": int(hist["n_backtrack_failures"]),
        "n_outer": int(hist["n_upper_iters"]),
        "f_trace": f_trace,
        "gd_trace": gd_trace,
        "x0": x_init,
        "xhat": x_final,
        "hist": hist,
    }


# %%
# Run both problems
# -----------------

problems = [make_inpainting(x_star), make_deblur(x_star)]
all_results = {}

for physics, y, name, detail in problems:
    print()
    print(f"=== {name} ({detail}), noise sigma={NOISE_SIGMA} ===", flush=True)
    print(f"y shape {tuple(y.shape)}", flush=True)
    theta0 = torch.tensor(THETA0, dtype=dtype, device=device)

    fixed = run_fixed(physics, y, x_star, theta0, MAX_OUTER, ALPHA0)
    print(
        f"fixed:  lam={fixed['lam']:.4f} f={fixed['f']:.4f} "
        f"PSNR {fixed['psnr0']:.2f} -> {fixed['psnr']:.2f} "
        f"GD={fixed['gd']} wall={fixed['wall']:.2f}s",
        flush=True,
    )

    maid = run_maid(
        physics, y, x_star, theta0, MAX_OUTER, ALPHA0, accelerated=False
    )
    print(
        f"MAID:   lam={maid['lam']:.4f} f={maid['f']:.4f} "
        f"PSNR {maid['psnr0']:.2f} -> {maid['psnr']:.2f} "
        f"GD={maid['gd']} BT={maid['bt']} wall={maid['wall']:.2f}s",
        flush=True,
    )

    acc = run_maid(
        physics, y, x_star, theta0, MAX_OUTER, ALPHA0, accelerated=True
    )
    print(
        f"accel:  lam={acc['lam']:.4f} f={acc['f']:.4f} "
        f"PSNR {acc['psnr0']:.2f} -> {acc['psnr']:.2f} "
        f"GD={acc['gd']} BT={acc['bt']} wall={acc['wall']:.2f}s",
        flush=True,
    )

    all_results[name] = {
        "physics": physics,
        "y": y,
        "detail": detail,
        "fixed": fixed,
        "maid": maid,
        "acc": acc,
    }


# %%
# Reconstruction figures
# ----------------------

for name, pack in all_results.items():
    y = pack["y"]
    maid = pack["maid"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    panels = [
        (x_star, "ground truth", None),
        (y, "measurement", None),
        (maid["x0"], f"init lambda=1\nPSNR {maid['psnr0']:.2f} dB", maid["psnr0"]),
        (
            maid["xhat"],
            f"MAID lambda={maid['lam']:.3f}\nPSNR {maid['psnr']:.2f} dB",
            maid["psnr"],
        ),
    ]
    for ax, (img, title, _) in zip(axes, panels):
        im = to_np(img)
        # Measurement for blur is still image-shaped; for display clamp.
        ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"{name}: Tikhonov weight learning (Set3C greyscale {IMG_SIZE}x{IMG_SIZE})",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / f"maid_imaging_{name}_recon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


# %%
# Convergence figures
# -------------------

for name, pack in all_results.items():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for key, label, style in [
        ("fixed", "fixed accuracy 1e-4", "-"),
        ("maid", "MAID", "-"),
        ("acc", "accelerated MAID", "-"),
    ]:
        r = pack[key]
        gd = r["gd_trace"]
        ftr = r["f_trace"]
        ax.plot(gd, ftr, style, label=label, linewidth=1.8)
    ax.set_xlabel("cumulative lower-level GD iterations")
    ax.set_ylabel("upper-level objective f(theta)")
    ax.set_title(f"{name}: objective vs lower-level work")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"maid_imaging_{name}_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


# %%
# Summary table
# -------------

print()
print(
    f"{'problem':<12} {'method':<18} {'PSNR0':>7} {'PSNR':>7} "
    f"{'lambda':>9} {'GD':>8} {'BT':>5} {'wall_s':>8}"
)
for name, pack in all_results.items():
    for key, label in [
        ("fixed", "fixed 1e-4"),
        ("maid", "MAID"),
        ("acc", "accelerated MAID"),
    ]:
        r = pack[key]
        print(
            f"{name:<12} {label:<18} {r['psnr0']:7.2f} {r['psnr']:7.2f} "
            f"{r['lam']:9.4f} {r['gd']:8d} {r['bt']:5d} {r['wall']:8.2f}"
        )

print()
print("Notes")
print(
    "  Prior: Tikhonov (TV weight IFT not wired; see module docstring)."
)
print(f"  Image: Set3C greyscale {IMG_SIZE}x{IMG_SIZE}, one supervised image.")
print(f"  Noise: Gaussian sigma={NOISE_SIGMA}.")
print(f"  Figures: {FIG_DIR}")
print(
    "  Numbers are measured as printed; if MAID does not beat fixed on a "
    "problem that fact is left in the table."
)
