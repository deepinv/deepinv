r"""
SkipProx vs unit-step PnP-PGD with a relaxed Prox-DRUNet denoiser.

Problem:   F_{lam,alpha}(x) = lam * f(x) + phi_{sigma,alpha}(x),  f(x) = 1/2 ||A x - y||_2^2 (sum of squares)
A:         circular convolution with the separable kernel [1,2,1]/4 x [1,2,1]/4 (deepinv BlurFFT),
           noise std 1/255, even-sized 256x256 RGB crops -> exact Fourier nullspace at the Nyquist frequency.
Denoiser:  D_{sigma,alpha}(v) = (1-alpha) v + alpha D_sigma(v), D_sigma = Prox-DRUNet, sigma = 5/255 fixed.

SkipProx round (one denoiser call, N = max(1, ceil(h)), h = lam * L_f):
    z = x ; repeat N: z = z - (lam * A^T(A z - y) + u) / N
    v = z + u ; x = D_alpha(v) ; u = v - x
PGD round (one denoiser call):  v = x - lam * A^T(A x - y) ; x = D_alpha(v) ; (u := v - x, logged only)

Diagnostic: r_k = ||lam A^T(A x_k - y) + u_k||_2 / sqrt(n)   (u_k = grad phi_{sigma,alpha}(x_k) for a proximal denoiser).

Subcommands
    prep     build the three ground truths and the fixed observations (CPU, seed 0) -> data/problem.pt
    checks   L_f from the Fourier spectrum, Fourier-vs-explicit inner loop, SkipProx(N=1) == PGD, sigma units,
             checkpoint provenance, local Jacobian estimates of Lip(I - D_sigma)
    sweep    run a grid of (algo, h, alpha) on one image, batched over configurations -> results/*.npz
    jacobian local Lip(I - D_sigma) estimates at final iterates of selected runs
    analyze  classify runs, write tables / heatmaps / curves / reconstruction figure (CPU)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import deepinv as dinv

HERE = Path(__file__).resolve().parent
SIGMA_Y = 1.0 / 255
SIGMA_DEN = 5.0 / 255
IMAGES = ["butterfly.png", "leaves.png", "CBSD_0010.png"]
CROP = 256
SEED = 0

# ---------------------------------------------------------------- tolerances (see README / report)
# float64 runs (GPU). In float32 the residual floor is ~3e-7 * lam (measured: 4.6e-6 at h=16), too coarse.
TOL_R = 1e-7  # stationarity residual: converged if max r_k over the last WIN rounds <= TOL_R ...
TOL_DX = 1e-7  # ... and max relative iterate change over the last WIN rounds <= TOL_DX
WIN = 20
DIV_XMAX = 1e3  # diverged if any |x| > DIV_XMAX or non-finite
CYC_P = 64  # lags tested for state repetition
CYC_TOL = 1e-6  # cycle at lag p>=2 if ||s_K - s_{K-p}|| / ||s_K|| <= CYC_TOL while r_K > TOL_R
CYC_RATIO = 0.05  # ... and it is < CYC_RATIO * the lag-1 distance (so it is not a slow fixed-point approach)


# =====================================================================================  problem
def kernel():
    k1 = torch.tensor([1.0, 2.0, 1.0]) / 4
    return (k1[:, None] * k1[None, :])[None, None]  # (1,1,3,3)


def make_physics(img_size, device, dtype=torch.float32):
    return dinv.physics.BlurFFT(
        img_size=img_size,
        filter=kernel().to(dtype),
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=SIGMA_Y),
    )


def ata_spectrum(physics, shape, device):
    """Real, non-negative spectrum a(k) of A^T A on the rfft2 grid, from deepinv's own A and A^T."""
    delta = torch.zeros(shape, device=device, dtype=physics.filter.dtype)
    delta[..., 0, 0] = 1.0
    h = physics.A_adjoint(physics.A(delta))
    s = torch.fft.rfft2(h)
    return s.real, s.imag.abs().max().item()


def cmd_prep(args):
    torch.manual_seed(SEED)
    gts = []
    for name in IMAGES:
        x = dinv.utils.load_example(name, img_size=None)
        H, W = x.shape[-2:]
        t, l = (H - CROP) // 2, (W - CROP) // 2
        gts.append(x[..., t : t + CROP, l : l + CROP].clone())
    gt = torch.cat(gts).float()
    phys = make_physics(gt.shape[1:], "cpu")
    g = torch.Generator().manual_seed(SEED)
    Ax = phys.A(gt)
    y = Ax + SIGMA_Y * torch.randn(Ax.shape, generator=g)
    out = dict(gt=gt, y=y, names=IMAGES, crop=CROP, seed=SEED, sigma_y=SIGMA_Y)
    (HERE / "data").mkdir(exist_ok=True)
    torch.save(out, HERE / "data" / "problem.pt")
    print("gt", gt.shape, gt.min().item(), gt.max().item(), "y", y.shape)
    for i, n in enumerate(IMAGES):
        print(n, "PSNR(y)", -10 * math.log10(((y[i] - gt[i]) ** 2).mean().item()))


def load_problem(device, dtype=torch.float32):
    p = torch.load(HERE / "data" / "problem.pt", map_location="cpu")
    return p["gt"].to(device, dtype), p["y"].to(device, dtype), p["names"]


# =====================================================================================  denoiser
class ProxDRUNet(torch.nn.Module):
    """D_sigma(x) = x - grad g_sigma(x) = N(x) + J_N(x)^T (x - N(x)), as deepinv.models.GSPnP.forward, but keeping the
    input dtype (GSPnP.potential_grad casts to float32) so that the whole iteration can run in float64."""

    def __init__(self, gs):
        super().__init__()
        self.gs = gs
        self.create_graph = False

    def forward(self, x, sigma):
        with torch.enable_grad():
            xx = x if self.create_graph else x.detach().requires_grad_(True)
            N = self.gs.student_grad(xx, sigma)
            JN = torch.autograd.grad(N, xx, grad_outputs=xx - N, create_graph=self.create_graph)[0]
        out = N + JN
        return out if self.create_graph else out.detach()


def load_denoiser(ckpt, device, dtype=torch.float32):
    gs = dinv.models.GSDRUNet(pretrained=ckpt, act_mode="s", device=device).to(device).eval()
    assert gs.alpha == 1.0
    for q in gs.parameters():
        q.requires_grad_(False)
    return ProxDRUNet(gs).to(device, dtype)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def relaxed(D, v, alpha, sigma=SIGMA_DEN):
    """D_{sigma,alpha}(v) = (1-alpha) v + alpha D_sigma(v); alpha is a (B,1,1,1) tensor."""
    return v + alpha * (D(v, sigma) - v)


# =====================================================================================  algorithms
class Fourier:
    """Exact N-step inner loop z <- z - (lam*(A^TA z - A^T y) + u)/N in rfft2 space."""

    def __init__(self, a, Aty):
        # (1,1,H,W//2+1), multipliers computed in float64 on CPU. The deepinv spectrum has |a| <= 1e-7 round-off
        # (slightly negative) on the Nyquist nullspace, clamp to the exact value 0.
        self.a = a.detach().cpu().double().clamp_min(0.0)
        self.dev = Aty.device
        self.bhat = torch.fft.rfft2(Aty)  # (1|B,C,H,W//2+1)

    def multipliers(self, lam, N):
        # q = 1 - lam a / N ; z_N = q^N z_0 + S/N * (lam b - u),  S = sum_{j<N} q^j
        lam = lam.detach().cpu().double().view(-1, 1, 1, 1)
        N = N.detach().cpu().double().view(-1, 1, 1, 1)
        q = 1 - lam * self.a / N
        qN = q**N
        S = torch.where((1 - q).abs() > 1e-12, (1 - qN) / (1 - q).clamp_min(1e-300), N)
        return qN.float().to(self.dev), (S / N).float().to(self.dev)

    def inner(self, x, u, lam, M1, M2):
        z = torch.fft.irfft2(
            M1 * torch.fft.rfft2(x) + M2 * (lam.view(-1, 1, 1, 1) * self.bhat - torch.fft.rfft2(u)),
            s=x.shape[-2:],
        )
        return z


def explicit_inner(phys, x, u, y, lam, N):
    z = x.clone()
    for _ in range(N):
        z = z - (lam * phys.A_adjoint(phys.A(z) - y) + u) / N
    return z


def run_batch(D, phys, four, gt, y, x0, algo, lam, alpha, K, check_rounds, log_every_state=False):
    """Run B configurations (same image) for K rounds. Returns logs dict."""
    B = lam.numel()
    dev = x0.device
    n = gt[0].numel()
    al = alpha.view(-1, 1, 1, 1)
    lamv = lam.view(-1, 1, 1, 1)
    hN = torch.clamp(torch.ceil(lam * 1.0 - 1e-9), min=1)  # L_f = 1 (checked), N = max(1, ceil(h))
    M1, M2 = four.multipliers(lam, hN)
    x = x0.expand(B, *x0.shape[1:]).clone()
    u = torch.zeros_like(x)
    yb = y.expand_as(x)
    grad_f = lambda z: phys.A_adjoint(phys.A(z) - yb)

    R = np.full((K, B), np.nan, np.float64)
    DX = np.full((K, B), np.nan, np.float64)
    FV = np.full((K, B), np.nan, np.float64)
    PS = np.full((K, B), np.nan, np.float64)
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    div_round = np.full(B, -1)
    ring = torch.zeros((CYC_P + 1, B, 2) + tuple(x.shape[1:]), device=dev)  # last CYC_P+1 states (x,u)
    cyc = {}
    best_ps = np.full(B, -np.inf)
    best_x = x.clone()
    for k in range(K):
        if not bool(alive.any()):
            break
        x_prev = x
        with torch.no_grad():
            if algo == "skip":
                z = four.inner(x, u, lam, M1, M2)
                v = z + u
            else:
                v = x - lamv * grad_f(x)
        v = torch.where(alive.view(-1, 1, 1, 1), v, torch.zeros_like(v))
        x = relaxed(D, v, al).detach()
        u = v - x
        with torch.no_grad():
            g = grad_f(x)
            r = (lamv * g + u).flatten(1).norm(dim=1) / math.sqrt(n)
            dx = (x - x_prev).flatten(1).norm(dim=1) / x_prev.flatten(1).norm(dim=1).clamp_min(1e-30)
            fv = 0.5 * (phys.A(x) - yb).flatten(1).pow(2).sum(1)
            ps = -10 * torch.log10((x - gt).flatten(1).pow(2).mean(1))
            bad = (~torch.isfinite(x).flatten(1).all(1)) | (x.flatten(1).abs().amax(1) > DIV_XMAX)
            newly = bad & alive
            if newly.any():
                for i in torch.nonzero(newly).flatten().tolist():
                    div_round[i] = k
            alive &= ~bad
            m = alive.cpu().numpy()
            for arr, t in ((R, r), (DX, dx), (FV, fv), (PS, ps)):
                arr[k] = np.where(m, t.float().cpu().double().numpy(), np.nan)
            x = torch.where(alive.view(-1, 1, 1, 1), x, torch.zeros_like(x))
            u = torch.where(alive.view(-1, 1, 1, 1), u, torch.zeros_like(u))
            imp = np.where(m, PS[k], -np.inf) > best_ps
            if imp.any():
                best_ps = np.where(imp, PS[k], best_ps)
                ii = torch.as_tensor(imp, device=dev).view(-1, 1, 1, 1)
                best_x = torch.where(ii, x, best_x)
            ring = torch.roll(ring, 1, dims=0)
            ring[0, :, 0] = x
            ring[0, :, 1] = u
            if (k + 1) in check_rounds and k + 1 > CYC_P:
                s0 = ring[0].flatten(1)
                d = torch.stack([(s0 - ring[p].flatten(1)).norm(dim=1) for p in range(1, CYC_P + 1)], 1)
                cyc[k + 1] = (d / s0.norm(dim=1, keepdim=True).clamp_min(1e-30)).cpu().numpy()  # (B, P)
    return dict(
        r=R, dx=DX, f=FV, psnr=PS, div_round=div_round, cyc=cyc,
        x_final=x.float().cpu().numpy(), x_best=best_x.half().cpu().numpy(),
        u_final=u.float().cpu().numpy(), N=hN.cpu().numpy(), rounds_run=k + 1,
    )


# =====================================================================================  checks
def lip_power(D, x, iters=30, seed=0):
    """Local estimate of ||J_{I - D_sigma}(x)||_2 (Hessian of g, symmetric) by power iteration with double backward."""
    D.create_graph = True
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(x.shape, generator=g).to(x.device, x.dtype)
    v = v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    est = None
    for _ in range(iters):
        xx = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            Rx = xx - D(xx, SIGMA_DEN)
            Hv = torch.autograd.grad(Rx, xx, grad_outputs=v)[0]
        est = Hv.flatten(1).norm(dim=1).detach()
        v = (Hv / est.view(-1, 1, 1, 1)).detach()
    D.create_graph = False
    return est.cpu().numpy()


def cmd_checks(args):
    dev, dt = args.device, getattr(torch, args.dtype)
    gt, y, names = load_problem(dev, dt)
    phys = make_physics(gt.shape[1:], dev, dt)
    rep = {"dtype": args.dtype, "device": str(dev), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    # ---- checkpoint provenance + units
    rep["ckpt_path"] = os.path.abspath(args.ckpt)
    rep["ckpt_sha256"] = sha256(args.ckpt)
    D = load_denoiser(args.ckpt, dev, dt)
    gen = torch.Generator().manual_seed(1)
    nz = torch.randn(gt.shape, generator=gen).to(dev, dt)
    # deepinv's own GSPnP forward (float32) vs the dtype-preserving wrapper
    with torch.no_grad():
        ref = load_denoiser(args.ckpt, dev, torch.float32).gs(gt.float() + 5 / 255 * nz.float(), 5 / 255)
        rep["wrapper_vs_GSPnP_forward_maxabs"] = (D(gt + 5 / 255 * nz, 5 / 255).float() - ref).abs().max().item()
    with torch.no_grad():
        pass
    units = {}
    for lab, s_in, s_noise in [("5/255 (unit range)", 5 / 255, 5 / 255), ("5 (0-255 range)", 5.0, 5 / 255),
                                ("25/255", 25 / 255, 25 / 255)]:
        with torch.no_grad():
            out = D(gt + s_noise * nz, s_in).detach()
        units[lab] = [float(-10 * torch.log10(((out[i] - gt[i]) ** 2).mean())) for i in range(len(names))]
    rep["denoise_psnr_by_sigma_arg"] = units
    # ---- L_f from the spectrum
    a, imag = ata_spectrum(phys, gt[:1].shape, dev)
    rep["L_f_fourier_max"] = a.max().item()
    rep["ata_min"] = a.min().item()
    rep["ata_imag_max"] = imag
    rep["n_zero_freqs(<1e-12)"] = int((a[0, 0] < 1e-12).sum().item())
    rep["L_f_deepinv_power"] = float(phys.compute_sqnorm(torch.rand_like(gt[:1]), max_iter=200, tol=1e-9, verbose=False)) \
        if hasattr(phys, "compute_sqnorm") else float(phys.compute_norm(torch.rand_like(gt[:1]), max_iter=200, tol=1e-9, verbose=False))
    # analytic: cos^2(pi k1/n) cos^2(pi k2/n)
    H, W = gt.shape[-2:]
    k1 = torch.arange(H, device=dev)[:, None]
    k2 = torch.arange(W // 2 + 1, device=dev)[None, :]
    ana = (torch.cos(math.pi * k1 / H) ** 4) * (torch.cos(math.pi * k2 / W) ** 4)
    rep["ata_vs_analytic_cos4_maxdiff"] = (a[0, 0] - ana).abs().max().item()
    # ---- Fourier inner loop vs explicit loop
    four = Fourier(a, phys.A_adjoint(y))
    x0 = phys.A_adjoint(y)
    gu = torch.Generator().manual_seed(2)
    u = 0.02 * torch.randn(gt.shape, generator=gu).to(dev, dt)
    agree = {}
    for h in [0.5, 1, 2, 3.7, 8, 16, 64]:
        lam = torch.full((len(names),), float(h), device=dev)
        N = torch.clamp(torch.ceil(lam - 1e-9), min=1)
        M1, M2 = four.multipliers(lam, N)
        zf = four.inner(x0, u, lam, M1, M2)
        ze = explicit_inner(phys, x0, u, y, float(h), int(N[0].item()))
        agree[str(h)] = dict(N=int(N[0]), max_abs=(zf - ze).abs().max().item(),
                             rel=((zf - ze).norm() / ze.norm()).item())
    rep["fourier_vs_explicit"] = agree
    # ---- SkipProx(N=1) == PGD, full trajectories (50 rounds, h in {0.5, 1}, alpha in {0.5, 1}), image 0
    eq = {}
    for h, al in [(0.5, 0.5), (1.0, 1.0), (1.0, 0.35)]:
        lam = torch.tensor([h], device=dev)
        alpha = torch.tensor([al], device=dev)
        o1 = run_batch(D, phys, Fourier(a, phys.A_adjoint(y[:1])), gt[:1], y[:1], x0[:1], "skip", lam, alpha, 50, set())
        o2 = run_batch(D, phys, Fourier(a, phys.A_adjoint(y[:1])), gt[:1], y[:1], x0[:1], "pgd", lam, alpha, 50, set())
        eq[f"h={h},alpha={al}"] = dict(
            max_abs_x_final=float(np.abs(o1["x_final"] - o2["x_final"]).max()),
            max_rel_r=float(np.nanmax(np.abs(o1["r"] - o2["r"]) / o2["r"])),
            max_abs_psnr=float(np.nanmax(np.abs(o1["psnr"] - o2["psnr"]))),
        )
    rep["skip_N1_vs_pgd_50rounds"] = eq
    for bs in [1, 8, 16]:
        z = gt[:1].expand(bs, -1, -1, -1).contiguous()
        D(z, SIGMA_DEN)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            D(z, SIGMA_DEN)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rep[f"sec_per_denoiser_call_batch{bs}"] = (time.time() - t) / 3
    # ---- local Jacobian estimates of Lip(I - D_sigma)
    if args.K == 0:
        print(json.dumps(rep, indent=1)); return
    rep["local_lip_I_minus_D_at_gt"] = lip_power(D, gt).tolist()
    rep["local_lip_I_minus_D_at_x0"] = lip_power(D, x0).tolist()
    rep["local_lip_I_minus_D_at_gt_plus_noise5"] = lip_power(D, gt + SIGMA_DEN * nz).tolist()
    print(json.dumps(rep, indent=1))
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(rep, open(HERE / "results" / "checks.json", "w"), indent=1)


# =====================================================================================  sweep
def cmd_sweep(args):
    dev, dt = args.device, getattr(torch, args.dtype)
    torch.manual_seed(SEED)
    gt_all, y_all, names = load_problem(dev, dt)
    i = args.image
    gt, y = gt_all[i : i + 1], y_all[i : i + 1]
    phys = make_physics(gt.shape[1:], dev, dt)
    a, _ = ata_spectrum(phys, gt.shape, dev)
    four = Fourier(a, phys.A_adjoint(y))
    x0 = phys.A_adjoint(y)
    D = load_denoiser(args.ckpt, dev, dt)
    hs = [float(v) for v in args.h.split(",")]
    als = [float(v) for v in args.alpha.split(",")]
    algos = args.algos.split(",")
    cfgs = [(al_, h, a_) for al_ in algos for h in hs for a_ in als]
    checks = {int(c) for c in args.check_rounds.split(",")} | {args.K}
    outdir = HERE / "results" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    for algo in algos:
        cs = [c for c in cfgs if c[0] == algo]
        for s in range(0, len(cs), args.batch):
            chunk = cs[s : s + args.batch]
            lam = torch.tensor([c[1] for c in chunk], device=dev, dtype=dt)  # L_f = 1 -> lam = h
            alpha = torch.tensor([c[2] for c in chunk], device=dev, dtype=dt)
            t0 = time.time()
            o = run_batch(D, phys, four, gt, y, x0, algo, lam, alpha, args.K, checks)
            fn = outdir / f"img{i}_{algo}_{args.part}{s:04d}.npz"
            np.savez_compressed(
                fn, h=lam.cpu().numpy(), alpha=alpha.cpu().numpy(), algo=algo, image=i, K=args.K, dtype=args.dtype,
                rounds_run=o["rounds_run"],
                r=o["r"], dx=o["dx"], f=o["f"], psnr=o["psnr"], div_round=o["div_round"], N=o["N"],
                cyc_rounds=np.array(sorted(o["cyc"])), cyc=np.stack([o["cyc"][k] for k in sorted(o["cyc"])]) if o["cyc"] else np.zeros(0),
                x_final=o["x_final"], x_best=o["x_best"], u_final=o["u_final"],
            )
            print(f"[{names[i]}] {algo} chunk {s}: {len(chunk)} cfgs x {o['rounds_run']} rounds in {time.time()-t0:.0f}s -> {fn.name}", flush=True)


# =====================================================================================  jacobian at final iterates
def cmd_jacobian(args):
    dev, dt = args.device, getattr(torch, args.dtype)
    D = load_denoiser(args.ckpt, dev, dt)
    out = []
    for fn in sorted((HERE / "results" / args.tag).glob("img*_*.npz")):
        z = np.load(fn)
        sel = [j for j in range(len(z["h"])) if np.isfinite(z["r"][-1, j])]
        if not sel:
            continue
        X = torch.from_numpy(z["x_final"][sel]).to(dev, dt)
        lips = np.concatenate([lip_power(D, X[s : s + 8], iters=args.iters) for s in range(0, len(sel), 8)])
        for j, l in zip(sel, lips):
            out.append(dict(file=fn.name, image=int(z["image"]), algo=str(z["algo"]), h=float(z["h"][j]),
                            alpha=float(z["alpha"][j]), local_lip=float(l)))
        print(fn.name, "local Lip(I-D) at x_K: min %.3f max %.3f" % (lips.min(), lips.max()), flush=True)
    json.dump(out, open(HERE / "results" / args.tag / "local_lip.json", "w"), indent=1)


# =====================================================================================  main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "checks", "sweep", "jacobian", "analyze"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.cache/torch/hub/checkpoints/Prox-DRUNet_torch.pth"))
    ap.add_argument("--image", type=int, default=0)
    ap.add_argument("--h", default="0.5,1,2,4,8,16,32,64")
    ap.add_argument("--alpha", default="0.1,0.25,0.35,0.5,0.75,1")
    ap.add_argument("--algos", default="skip,pgd")
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--check_rounds", default="200,1000")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="screen")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--dtype", default="float64" if torch.cuda.is_available() else "float32")
    ap.add_argument("--part", default="", help="file-name prefix to keep several sweeps of one image apart")
    args = ap.parse_args()
    if args.cmd == "analyze":
        from analyze import main as analyze_main

        analyze_main(args)
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 would silently cut float32 convs to 10-bit mantissas
        torch.backends.cudnn.allow_tf32 = False
        globals()["cmd_" + args.cmd](args)
