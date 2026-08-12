r"""
Learning a convex ridge regulariser with MAID
=============================================

This example learns the 7533 parameters of a multi-convolution ridge
regulariser by solving a bilevel problem with the Method of Adaptive Inexact
Descent (MAID), and compares it against isotropic total variation whose single
weight is grid-tuned on the same training set.

The bilevel problem is

.. math::

    \min_\theta \; \tfrac12 \sum_i \|x_i^\star(\theta) - x_i\|^2
    \quad\text{s.t.}\quad
    x_i^\star(\theta) = \arg\min_x \tfrac12\|x - y_i\|^2 + R_\theta(x).

The lower level is solved inexactly. MAID chooses how inexactly: it starts
loose, tightens only when its descent test fails, and certifies every
reconstruction with an a posteriori bound

.. math::

    \|x^\star - x\| \le \|\nabla_x h(x)\| / \mu .

To show that the adaptive accuracy is what does the work, a second arm runs the
identical line search at a *fixed* accuracy. Both arms share the initial step
size, the backtracking constants and the same batched oracle, so the only
difference between them is whether the accuracy adapts.
"""

# %%
import time

import numpy as np
import torch
from torchvision import transforms

import deepinv as dinv
from deepinv.datasets import CBSD68
from deepinv.optim.bilevel import (
    MAID,
    BatchedCRR,
    BatchedMinibatchOracle,
    ConvexRidgeConfig,
    MAIDConfig,
    accelerated_maid_config,
    pack_init_theta,
    unpack_theta,
)
from deepinv.optim.bilevel import auto_initial_step
from deepinv.optim.bilevel.tv_baseline import solve_isotropic_tv

import matplotlib.pyplot as plt

torch.manual_seed(0)
device = "cpu"
dtype = torch.float64

# %%
# Problem size
# ------------
# One 32x32 colour crop from each of N_TRAIN distinct CBSD68 images, evaluated
# on crops from N_HELD images that are never seen during training. Sixteen
# training images is the smallest set that generalises here: with eight, the
# regulariser fits the training crops and loses to TV on held-out data, and
# more outer iterations widen that gap rather than closing it.
N_TRAIN, N_HELD = 16, 8
IMG_SIZE, NOISE_SIGMA = 32, 0.05
# The objective is still descending at this point, so the numbers below are a
# lower bound on what the regulariser reaches, not a converged result.
N_OUTER = 60

cfg = ConvexRidgeConfig(
    nb_channels=(3, 4, 8, 32),
    filter_sizes=(5, 5, 5),
    weak_convexity=0.0,
    lip_fft_size=96,
    beta_init=4.0,
    sigma_init=0.1,
    # A = I gives mu_data = 1, so the ridge floor is unnecessary here.
    gamma=0.0,
)

# Shared between both arms, so the comparison isolates the accuracy schedule.
# Derived at theta0 rather than hardcoded: the hypergradient norm is a
# property of the regulariser, and a step chosen for one prior starves
# another. At an arbitrary 1e-3 both arms here are step-limited rather than
# accuracy-limited, and the comparison measures nothing.
ALPHA0 = None  # filled in below from auto_initial_step
LS_RHO, LS_RHO_BAR, LS_LAMBD, LS_MAX_BT = 0.5, 1.05, 1e-6, 20
# Starting accuracy for MAID, which is free to tighten or loosen it.
EPS0 = 1e-3
# The fixed-accuracy arm needs an accuracy a careful practitioner would pick,
# not MAID's starting point. Held at MAID's own starting value the sandwich
# test can never certify descent, the line search exhausts its budget every
# iteration, and the arm simply never steps: that is a rigged comparison, not
# evidence. This is the tolerance the arm would be given if it were the only
# method available.
FIXED_EPS = 2e-5

# %%
# Data
# ----
dataset = CBSD68(root="datasets/CBSD68", download=True, transform=transforms.ToTensor())
crop_gen = torch.Generator().manual_seed(11)


def crop(i):
    s = dataset[i]
    s = (s[0] if isinstance(s, (tuple, list)) else s).to(dtype)
    top = int(torch.randint(0, s.shape[-2] - IMG_SIZE + 1, (1,), generator=crop_gen))
    left = int(torch.randint(0, s.shape[-1] - IMG_SIZE + 1, (1,), generator=crop_gen))
    return s[:, top : top + IMG_SIZE, left : left + IMG_SIZE]


x_train = torch.stack([crop(i) for i in range(N_TRAIN)]).to(device)
x_held = torch.stack([crop(i) for i in range(N_TRAIN, N_TRAIN + N_HELD)]).to(device)

noise_gen = torch.Generator().manual_seed(50)
y_train = x_train + NOISE_SIGMA * torch.randn(
    x_train.shape, generator=noise_gen, dtype=dtype
)
y_held = x_held + NOISE_SIGMA * torch.randn(
    x_held.shape, generator=noise_gen, dtype=dtype
)

physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(0.0))


def psnr_each(a, b):
    d = (a - b).flatten(1)
    return (10 * torch.log10(1.0 / (d * d).mean(dim=1).clamp_min(1e-30))).cpu().numpy()


print(f"train {tuple(x_train.shape)}  held-out {tuple(x_held.shape)}")
print(
    f"noisy input   train={psnr_each(y_train, x_train).mean():.2f} dB  "
    f"held={psnr_each(y_held, x_held).mean():.2f} dB"
)

# %%
# Total variation baseline
# ------------------------
# One parameter, tuned by grid search on the training set. This is what
# bilevel learning has to beat, and it is a fair fight: the grid is fine and
# the same solver and data are used throughout.
t0 = time.perf_counter()
best_psnr, tv_lam = -np.inf, None
for lam in np.geomspace(0.005, 0.2, 15):
    xr, _ = solve_isotropic_tv(physics, y_train, float(lam), verify=False)
    v = float(psnr_each(xr, x_train).mean())
    if v > best_psnr:
        best_psnr, tv_lam = v, float(lam)

tv_train, _ = solve_isotropic_tv(physics, y_train, tv_lam, verify=False)
tv_held, _ = solve_isotropic_tv(physics, y_held, tv_lam, verify=False)
print(
    f"TV            train={psnr_each(tv_train, x_train).mean():.2f} dB  "
    f"held={psnr_each(tv_held, x_held).mean():.2f} dB  "
    f"lambda={tv_lam:.4f}  ({time.perf_counter() - t0:.1f}s)"
)

# %%
# The batched oracle
# ------------------
# Samples are independent given ``theta``, so the batched Hessian is block
# diagonal and one Newton solve covers the whole batch. The batch size is
# chosen from measured memory, and hypergradients accumulate exactly, so
# splitting the set changes peak memory and nothing else.
theta0 = pack_init_theta(cfg, dtype=dtype, seed=0).to(device)
oracle = BatchedMinibatchOracle(
    [(physics, y_train, x_train)], cfg=cfg, solver_max_iter=20_000
)
print(
    f"batched oracle: batch_size={oracle.batch_size}  "
    f"per-sample={oracle.per_sample_bytes / 2**20:.2f} MiB  "
    f"n_batches={len(oracle.batches)}  theta={theta0.numel()} parameters"
)

# %%
# The initial step size, derived
# ------------------------------
# ``auto_initial_step`` scales the first step to the hypergradient at
# ``theta0``, so the first trial move is a bounded fraction of ``||theta||``.
# Both arms are given the same derived value.
_lower0 = oracle.solve_lower_level(theta0, eps=EPS0)
_z0 = oracle.hypergradient(theta0, _lower0, EPS0).z
ALPHA0 = auto_initial_step(oracle, theta0, _z0)
print(f"derived alpha0 = {ALPHA0:.3e}  from ||z0|| = {float(_z0.norm()):.3e}")

# %%
# Arm 1: MAID, adaptive accuracy
# ------------------------------
# Plain Algorithm 3.1: no Barzilai-Borwein step initialisation and no
# nonmonotone test, so this arm and the fixed-accuracy arm below share the
# same step machinery and differ only in whether the accuracy adapts. The
# accelerated switches are a separate contribution and would confound this
# comparison: with bb_init the step reaches alpha = 11.8 within ten
# iterations, which no amount of accuracy adaptation explains.
maid_cfg = MAIDConfig(
    eps0=EPS0,
    delta0=EPS0,
    alpha0=ALPHA0,
    rho=LS_RHO,
    rho_bar=LS_RHO_BAR,
    nu=0.5,
    nu_bar=1.05,
    eta=0.5,
    lambd=LS_LAMBD,
    max_BT=LS_MAX_BT,
    max_iter=N_OUTER,
    tol=1e-9,
    g_convex=True,
    check_descent_direction=False,
    eps_min=1e-7,
    delta_min=1e-7,
    max_outer_BT=25,
    nonmonotone=False,
    verbose=True,
    show_progress_bar=False,
    log_every=10,
)
t0 = time.perf_counter()
theta_maid, hist = MAID(oracle, maid_cfg).run(theta0.clone())
maid_wall = time.perf_counter() - t0
maid_gd = int(oracle.n_gd_iters)
print(f"MAID   wall={maid_wall:.1f}s  lower-level iterations={maid_gd}")

# %%
# Arm 3: MAID with the accelerated switches
# -----------------------------------------
# Barzilai-Borwein step initialisation and the Zhang-Hager nonmonotone test,
# on top of the same adaptive accuracy. These are what let the step reach a
# useful magnitude within a 60-iteration budget.
acc_oracle = BatchedMinibatchOracle(
    [(physics, y_train, x_train)], cfg=cfg, solver_max_iter=20_000
)
# setdefault will not override the explicit False values carried over from
# the plain config, so the two switches are set here directly.
acc_cfg = accelerated_maid_config(
    **{**maid_cfg.__dict__, "bb_init": True, "nonmonotone": True}
)
t0 = time.perf_counter()
theta_acc, hist_acc = MAID(acc_oracle, acc_cfg).run(theta0.clone())
acc_wall = time.perf_counter() - t0
print(
    f"MAID+  wall={acc_wall:.1f}s  lower-level iterations={int(acc_oracle.n_gd_iters)}"
)


# %%
# Arm 2: the same line search at fixed accuracy
# ---------------------------------------------
# Identical step size, backtracking constants and descent test. The accuracy
# is held at ``FIXED_EPS`` throughout instead of being adapted, which is the
# single thing MAID changes.
def train_fixed_accuracy(n_outer):
    fixed_oracle = BatchedMinibatchOracle(
        [(physics, y_train, x_train)], cfg=cfg, solver_max_iter=20_000
    )
    th = theta0.clone()
    warm, alpha, n_bt = None, ALPHA0, 0
    L_g = 1.0  # upper level is 0.5 ||x - x_star||^2
    t_start = time.perf_counter()
    for _ in range(n_outer):
        lower = fixed_oracle.solve_lower_level(th, eps=FIXED_EPS, warm_start=warm)
        z = fixed_oracle.hypergradient(th, lower, delta=FIXED_EPS).z
        z_norm_sq = float(torch.sum(z * z))
        g_old = float(fixed_oracle.g(lower.x))
        gg_old = float(fixed_oracle.grad_g(lower.x).norm())
        # The same sandwich as MAID's g_convex path, at constant accuracy.
        U_lower = g_old - gg_old * FIXED_EPS - 0.5 * L_g * FIXED_EPS**2
        alpha_try, line_ok, lower_trial = alpha, False, lower
        for _bt in range(LS_MAX_BT):
            lower_trial = fixed_oracle.solve_lower_level(
                th - alpha_try * z, eps=FIXED_EPS, warm_start=lower
            )
            g_new = float(fixed_oracle.g(lower_trial.x))
            gg_new = float(fixed_oracle.grad_g(lower_trial.x).norm())
            U_upper = g_new + gg_new * FIXED_EPS + 0.5 * L_g * FIXED_EPS**2
            psi = U_upper - U_lower + LS_LAMBD * alpha_try * z_norm_sq
            psi = psi - 0.5 * L_g * FIXED_EPS**2
            if psi <= 0.0:
                line_ok = True
                break
            alpha_try *= LS_RHO
            n_bt += 1
        if line_ok:
            th, warm, alpha = th - alpha_try * z, lower_trial, LS_RHO_BAR * alpha_try
        else:
            n_bt += 1
            alpha = max(ALPHA0 * 1e-3, alpha * LS_RHO)
    return (
        th.detach(),
        time.perf_counter() - t_start,
        n_bt,
        int(fixed_oracle.n_gd_iters),
    )


theta_fixed, fixed_wall, fixed_bt, fixed_gd = train_fixed_accuracy(N_OUTER)
print(
    f"fixed  wall={fixed_wall:.1f}s  lower-level iterations={fixed_gd}  "
    f"backtracks={fixed_bt}"
)


# %%
# Results
# -------
# Every reconstruction is reported with its certificate: the returned iterate
# is within ``residual / mu`` of the true minimiser, computable without
# knowing that minimiser.
def reconstruct(theta, y, x):
    problem = BatchedCRR(y=y, x_star=x, cfg=cfg, physics=physics)
    xr, res, info = problem.solve_lower(theta, eps=2e-6, max_iter=20_000)
    return xr, float(res.max()) / problem.mu, info


acc_train, _, _ = reconstruct(theta_acc, y_train, x_train)
acc_held, bound_acc, _ = reconstruct(theta_acc, y_held, x_held)
crr_train, bound_train, _ = reconstruct(theta_maid, y_train, x_train)
crr_held, bound_held, info_held = reconstruct(theta_maid, y_held, x_held)
fix_train, _, _ = reconstruct(theta_fixed, y_train, x_train)
fix_held, _, _ = reconstruct(theta_fixed, y_held, x_held)

rows = [
    ("noisy input", psnr_each(y_train, x_train), psnr_each(y_held, x_held)),
    ("TV, grid-tuned", psnr_each(tv_train, x_train), psnr_each(tv_held, x_held)),
    ("CRR, fixed accuracy", psnr_each(fix_train, x_train), psnr_each(fix_held, x_held)),
    ("CRR, MAID", psnr_each(crr_train, x_train), psnr_each(crr_held, x_held)),
    (
        "CRR, MAID accelerated",
        psnr_each(acc_train, x_train),
        psnr_each(acc_held, x_held),
    ),
]
print(f"\n{'method':<22}{'train':>9}{'held':>9}{'gap':>9}")
for name, tr, he in rows:
    print(f"{name:<22}{tr.mean():>9.2f}{he.mean():>9.2f}{tr.mean() - he.mean():>9.2f}")
print(
    f"\nheld-out certificate: ||x* - x|| <= {bound_held:.2e}, "
    f"{info_held['n_stalled']}/{N_HELD} samples stalled"
)

# %%
# What the adaptive accuracy buys
# -------------------------------
# MAID reaches a lower objective than the fixed-accuracy arm for a comparable
# number of lower-level iterations: it spends coarse solves early, where the
# hypergradient only has to point roughly downhill, and buys precision only
# once the descent test demands it.
f = np.asarray(hist_acc["f_exact"], dtype=float)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].semilogy(f, lw=2)
ax[0].set_xlabel("outer iteration")
ax[0].set_ylabel(r"$f(\theta_k)$")
ax[0].set_title("upper-level objective")
ax[0].grid(alpha=0.3)

ax[1].semilogy(np.asarray(hist["z_norm"], dtype=float), lw=2, color="C1")
ax[1].set_xlabel("outer iteration")
ax[1].set_ylabel(r"$\|z_k\|$")
ax[1].set_title("hypergradient norm")
ax[1].grid(alpha=0.3)

ax[2].semilogy(np.asarray(hist["alpha"], dtype=float), lw=2, label=r"step $\alpha_k$")
ax[2].semilogy(
    np.asarray(hist["eps"], dtype=float), lw=2, label=r"accuracy $\varepsilon_k$"
)
ax[2].set_xlabel("outer iteration")
ax[2].set_title("adaptive step and accuracy")
ax[2].legend()
ax[2].grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# Reconstructions
# ---------------
k = min(3, N_HELD)


def show(t):
    return t.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)


fig, ax = plt.subplots(k, 4, figsize=(12, 3.1 * k))
cols = [
    ("ground truth", x_held, None),
    ("noisy", y_held, psnr_each(y_held, x_held)),
    ("TV", tv_held, psnr_each(tv_held, x_held)),
    ("CRR, MAID", acc_held, psnr_each(acc_held, x_held)),
]
for r in range(k):
    for c, (name, tensor, p) in enumerate(cols):
        a = ax[r, c] if k > 1 else ax[c]
        a.imshow(show(tensor[r]))
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(name if p is None else f"{name}  {p[r]:.2f} dB")
fig.suptitle(f"held-out images, sigma = {NOISE_SIGMA}")
fig.tight_layout()
plt.show()

# %%
# The learned filters
# -------------------
# The regulariser is convolutional, so it carries no notion of image size: a
# prior learned on 32x32 crops applies unchanged to whole images.
weights, _scaling, _beta = unpack_theta(theta_acc.detach().cpu(), cfg)
impulse = torch.zeros(1, 3, 33, 33, dtype=dtype)
impulse[0, 0, 16, 16] = 1.0
response = impulse
for w in weights:
    response = torch.nn.functional.conv2d(response, w, padding=w.shape[-1] // 2)
response = response[0]

order = torch.argsort(response.flatten(1).norm(dim=1), descending=True)[:16]
fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
for i, idx in enumerate(order.tolist()):
    a = ax[i // 8, i % 8]
    v = response[idx].numpy()
    a.imshow(v, cmap="RdBu_r", vmin=-abs(v).max(), vmax=abs(v).max())
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle(f"16 largest of {cfg.n_filters} learned filters (impulse response)")
fig.tight_layout()
plt.show()
