r"""
Cryo-electron tomography
========================

Simulate and reconstruct cryo-electron tomography (cryo-ET) data with :class:`deepinv.physics.CryoET`.

We first reconstruct an experimental tilt series with the weighted back-projection (WBP) :meth:`deepinv.physics.CryoET.fbp`, check its consistency by re-projection, and compare it with the deep-learning reconstruction :class:`deepinv.models.CryoLithe`.
We then simulate tilt series with contrast transfer function (CTF), defocus gradient, dose weighting, ice thickness, shot noise and deformations.

The data is an aligned tilt series of 41 tilts of size :math:`1024 \times 1024` (pixel size 7.84 Å) acquired in :math:`[-72^\circ, 48^\circ]`.
By default, we use a version binned by 4 stored as a NumPy archive. The full-resolution data and its reference WBP and Icecream reconstructions, stored as MRC files, can be used by setting `CRYOET_DATA` to the tilt series MRC file. If unavailable, a synthetic phantom is used.
"""

# %%

# sphinx_gallery_tags = ["Tomography", "Microscopy"]

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import deepinv as dinv
from deepinv.physics.functional import ctf, dose_filter

device = dinv.utils.get_device()
if device.type == "mps":  # 3D grid sampling backward is not implemented on MPS
    device = torch.device("cpu")
torch.manual_seed(0)
generator = torch.Generator(device).manual_seed(0)

PREFIX = "06022023_BrnoKrios_Arctis_xe_Position_70"
PIXEL_SIZE = 7.84  # Å
DATA_PATH = Path(os.environ.get("CRYOET_DATA", f"tilt_series_{PREFIX}_bin4.npz"))
DATA_DIR = DATA_PATH.parent
use_real_data = DATA_PATH.exists()
use_mrc = use_real_data and DATA_PATH.suffix == ".mrc"

# MRC files (full-resolution tilt series, with angles and reference reconstructions in the same folder) require mrcfile
if use_mrc:
    try:
        import mrcfile
    except ImportError as e:
        raise ImportError(
            "Reading MRC files requires mrcfile: pip install mrcfile"
        ) from e


def read_mrc(name, factor=1, crop=(slice(None),) * 3, bin_depth=True):
    r"""Reads a cropped volume or tilt series from a memory-mapped MRC file, binned by `factor`."""
    step = factor if bin_depth else 1
    with mrcfile.mmap(DATA_DIR / f"{name}.mrc", permissive=True) as mrc:
        data = mrc.data[crop[0]]
        slabs = [
            torch.from_numpy(data[i : i + step, crop[1], crop[2]].astype("float32"))
            for i in range(0, data.shape[0] - step + 1, step)
        ]
    return F.avg_pool2d(torch.stack([s.mean(0) for s in slabs]), factor)


def make_phantom(shape, n_vesicles, n_particles, seed):
    r"""Synthetic phantom of membrane vesicles (Gaussian shells) and macromolecules (Gaussian blobs)."""
    g = torch.Generator().manual_seed(seed)
    grid = torch.stack(
        torch.meshgrid(
            *[torch.arange(n, dtype=torch.float32) for n in shape], indexing="ij"
        ),
        -1,
    )
    vol = torch.zeros(shape)
    for i in range(n_vesicles + n_particles):
        dist = (grid - torch.rand(3, generator=g) * torch.tensor(shape)).norm(dim=-1)
        r = torch.rand(1, generator=g)
        if i < n_vesicles:
            vol += torch.exp(-((dist - (0.08 + 0.12 * r) * shape[-1]) ** 2) / 2)
        else:
            vol += 0.7 * torch.exp(-(dist**2) / (2 * (1 + 1.5 * r) ** 2))
    return vol


# %%
# Reconstruction of experimental data
# -----------------------------------
#
# We bin the data by 8 (pixel size 62.7 Å), giving a :math:`41 \times 128 \times 128` tilt series and a :math:`64 \times 128 \times 128` volume.
# Tilts and volumes are stored as `(N, H, W)` and `(D, H, W)` with the tilt axis along `H`, as in :class:`deepinv.physics.CryoET`.
#
# Under the weak-phase approximation, tilt :math:`i` is :math:`I_i = s_i (1 + h_i * P_{\theta_i} x)`. We use the relative contrast :math:`I_i / \bar I_i - 1`, which removes the factors :math:`s_i` (hence `ice_thickness=0`).
# We use a nominal defocus of 4 µm. At this pixel size, the CTF is negative at all frequencies, so that phase flipping inverts the contrast, and radiation damage only affects frequencies above Nyquist, so we disable dose weighting.

bin_factor = 8
if use_mrc:
    tilts = read_mrc(DATA_PATH.stem, bin_factor, bin_depth=False)
    angles = torch.tensor(
        [float(a) for a in (DATA_DIR / f"angles_{PREFIX}.tlt").read_text().split()]
    )
elif use_real_data:
    data = np.load(DATA_PATH)
    factor = round(PIXEL_SIZE * bin_factor / float(data["pixel_size"]))
    tilts = F.avg_pool2d(torch.from_numpy(data["tilts"]).float(), factor)
    angles = torch.from_numpy(data["angles"])
if use_real_data:
    vol_shape = (tilts.shape[-1] // 2, *tilts.shape[-2:])
else:
    vol_shape = (64, 128, 128)
    angles = torch.arange(-60.0, 61.0, 3.0)
    x0 = make_phantom(vol_shape, n_vesicles=12, n_particles=150, seed=0)[None, None]
    sim = dinv.physics.CryoET(
        img_size=vol_shape,
        angles=angles,
        pixel_size=PIXEL_SIZE * bin_factor,
        defocus=4.0,
        background=1.0,
    )
    sim.noise_model = dinv.physics.PoissonNoise(
        gain=1 / sim.electrons_per_pixel, clip_positive=True
    )
    tilts = sim(0.05 * x0 / sim.A(x0).std())[0, 0]

y = (tilts / tilts.mean(dim=(-1, -2), keepdim=True) - 1)[None, None].to(device)
angles = angles.to(device)
n_tilts = len(angles)
i_lo, i0, i_hi = 0, int(torch.argmin(angles.abs())), n_tilts - 1

physics = dinv.physics.CryoET(
    img_size=vol_shape,
    angles=angles,
    pixel_size=PIXEL_SIZE * bin_factor,
    defocus=4.0,
    dose_weighting=False,
    ice_thickness=0.0,
    device=device,
)

dinv.utils.plot(
    [y[:, :, i] for i in (i_lo, i0, i_hi)],
    titles=[f"{angles[i].item():.0f}°" for i in (i_lo, i0, i_hi)],
    suptitle="Measured tilts (relative contrast)",
)

# %%
# Weighted back-projection
# ~~~~~~~~~~~~~~~~~~~~~~~~
# :meth:`deepinv.physics.CryoET.fbp` phase-flips the tilts, ramp-filters them perpendicularly to the tilt axis, weights them by their angular spacing and back-projects them.
# We compare it with the reference reconstructions, whose contrast is inverted with respect to ours.

x_fbp = physics.fbp(y)

volumes, titles = [x_fbp], ["WBP (deepinv)"]
if use_mrc:
    x_ref = -read_mrc(f"vol_{PREFIX}_split1_fbp_float16", bin_factor)[None, None].to(
        device
    )
    x_icecream = -read_mrc(f"vol_{PREFIX}_Icecream", bin_factor)[None, None].to(device)
    volumes += [x_ref, x_icecream]
    titles += ["WBP (reference, half dose)", "Icecream (reference)"]
    corr = torch.corrcoef(torch.stack([x_fbp.flatten(), x_ref.flatten()]))[0, 1]
    print(f"Correlation between our WBP and the reference WBP: {corr:.3f}")
elif not use_real_data:
    volumes += [x0.to(device)]
    titles += ["Ground truth"]

dinv.utils.plot_ortho3D(volumes, titles=titles, figsize=(4 * len(volumes), 5))

# %%
# Re-projection
# ~~~~~~~~~~~~~
# We re-project the WBP and a least-squares reconstruction (10 conjugate gradient iterations of :meth:`deepinv.physics.CryoET.A_dagger`), and compute their per-tilt correlation with the measurements.
# As the WBP ignores the CTF amplitude, we fit a global scale. As the ramp filter removes the lowest frequencies, we compare high-pass filtered tilts.

blur = dinv.physics.Blur(
    filter=dinv.physics.blur.gaussian_blur(6.0), padding="reflect", device=device
)


def high_pass(t, margin=8):
    flat = t.flatten(0, 2)[:, None]
    return (flat - blur(flat)).reshape(t.shape)[..., margin:-margin, margin:-margin]


def correlation(a, b):
    a, b = a - a.mean((-1, -2), True), b - b.mean((-1, -2), True)
    return (
        (a * b).sum((-1, -2)) / (a.norm(dim=(-1, -2)) * b.norm(dim=(-1, -2)))
    ).flatten()


y_fbp = physics.A(x_fbp)
y_ls = physics.A(physics.A_dagger(y, solver="CG", max_iter=10))
corr_fbp = correlation(high_pass(y), high_pass(y_fbp))
corr_ls = correlation(high_pass(y), high_pass(y_ls))
print(
    f"Mean re-projection correlation: WBP {corr_fbp.mean():.3f}, least squares {corr_ls.mean():.3f}"
)

scale = (high_pass(y) * high_pass(y_fbp)).sum() / high_pass(y_fbp).pow(2).sum()
dinv.utils.plot(
    [
        high_pass(y)[:, :, i0],
        scale * high_pass(y_fbp)[:, :, i0],
        (high_pass(y) - scale * high_pass(y_fbp))[:, :, i0],
    ],
    titles=["measured", "WBP re-projection", "residual"],
    suptitle=f"High-pass filtered tilt at {angles[i0]:.0f}°",
)

ang = angles.cpu()
plt.figure(figsize=(5, 3.5))
plt.plot(ang, corr_fbp.cpu(), "o-", markersize=3, label="WBP")
plt.plot(ang, corr_ls.cpu(), "o-", markersize=3, label="Least squares")
plt.xlabel("tilt angle (°)")
plt.title("Re-projection correlation")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# The agreement degrades at high tilts, where the electron path is longer and, in a dose-symmetric scheme, radiation damage is larger.
# It also drops near :math:`0^\circ`, also for the reference reconstructions: these tilts are acquired first, when beam-induced motion is strongest, which can be modelled with the deformations of the operator (see below).

# %%
# Deep-learning reconstruction with CryoLithe
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :class:`deepinv.models.CryoLithe` :footcite:p:`kishore2025cryolithe` is a supervised network estimating a denoised, missing-wedge-corrected volume from an aligned tilt series (see the `original code <https://github.com/swing-research/CryoLithe>`_).
# Each wavelet coefficient of the volume is estimated by an MLP from :math:`21 \times 21` patches of the ramp-filtered tilts around its projections. Pretrained weights are downloaded from Hugging Face.
#
# As patches are local, the network must be applied at its training pixel size (about 8 Å), so we use raw full-resolution tilts if available (MRC data). On binned tilts, the result is only indicative.
# As voxels are estimated independently, we reconstruct a region of interest only. The output keeps the contrast of the tilts, which we invert.
#
# .. warning::
#
#     This tilt series is part of the CryoLithe training set, so this result is not representative of the performance on unseen data.

nn_device = dinv.utils.get_device(verbose=False)
cryolithe = dinv.models.CryoLithe(
    pretrained="download",
    variant="cryolithe",
    dtype=torch.float16 if nn_device.type == "cuda" else torch.float32,
    device=nn_device,
)

if use_mrc:
    tilts_full = read_mrc(DATA_PATH.stem, bin_depth=False)
    depth_full = 512
    roi = (slice(208, 304), slice(416, 608), slice(416, 608))
else:
    tilts_full, depth_full, roi = tilts, vol_shape[0], None

x_cryolithe = -cryolithe(
    tilts_full[None, None].to(nn_device),
    angles=angles.to(nn_device),
    depth=depth_full,
    roi=roi,
).to(device)

volumes, titles = [x_cryolithe], ["CryoLithe"]
if use_mrc:
    volumes += [
        -read_mrc(f"vol_{PREFIX}_{name}", 1, roi)[None, None].to(device)
        for name in ("split1_fbp_float16", "Icecream")
    ]
    titles += ["WBP (reference, half dose)", "Icecream (reference)"]
    corr = torch.corrcoef(torch.stack([x_cryolithe.flatten(), volumes[-1].flatten()]))[
        0, 1
    ]
    print(f"Correlation between CryoLithe and Icecream: {corr:.3f}")
else:
    volumes += [x_fbp] if use_real_data else [x_fbp, x0.to(device)]
    titles += ["WBP"] if use_real_data else ["WBP", "Ground truth"]
dinv.utils.plot_ortho3D(volumes, titles=titles, figsize=(4 * len(volumes), 5))

# %%
# Simulations
# -----------
#
# We use a crop of the full-resolution Icecream reconstruction (MRC data) or a synthetic phantom as ground truth, with inverted contrast so that macromolecules have positive potential, tapered at the borders and scaled to a typical projected contrast of 5%.

if use_mrc:
    x_gt = -read_mrc(
        f"vol_{PREFIX}_Icecream", 1, (slice(128, 384), slice(448, 576), slice(448, 576))
    )
else:
    x_gt = make_phantom((256, 128, 128), n_vesicles=4, n_particles=400, seed=1)
    angles = torch.arange(-60.0, 61.0, 3.0, device=device)
    n_tilts = len(angles)
    i_lo, i0, i_hi = 0, int(torch.argmin(angles.abs())), n_tilts - 1
    ang = angles.cpu()

sim_shape = tuple(x_gt.shape)
ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, 16))
tapers = [torch.cat([ramp, torch.ones(n - 32), ramp.flip(0)]) for n in sim_shape]
x_gt = (
    (x_gt - x_gt.mean())
    * tapers[0][:, None, None]
    * tapers[1][None, :, None]
    * tapers[2]
)
x_gt = x_gt[None, None].to(device)

common = dict(
    img_size=sim_shape,
    angles=angles,
    pixel_size=PIXEL_SIZE,
    fft_padding=16,
    device=device,
)
x_gt = x_gt * 0.05 / dinv.physics.CryoET(defocus=3.0, **common).A(x_gt).std()

# %%
# Contrast transfer function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# The :func:`CTF <deepinv.physics.functional.ctf>` oscillates faster for larger defocus, which improves low-resolution contrast but attenuates and inverts high frequencies.
# The :func:`dose filter <deepinv.physics.functional.dose_filter>` attenuates high frequencies as the accumulated exposure increases.

k = torch.linspace(0, 1 / (2 * PIXEL_SIZE), 500)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
for df in (2.0, 5.0):
    axs[0].plot(k, ctf(k, torch.zeros_like(k), defocus=df), label=f"defocus {df} µm")
for exposure in (0.0, 20.0, 60.0, 120.0):
    axs[1].plot(k, dose_filter(k, exposure), label=f"{exposure:.0f} e/Å²")
axs[0].set(xlabel="spatial frequency (1/Å)", title="CTF")
axs[1].set(xlabel="spatial frequency (1/Å)", title="Dose filter")
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()

ideal = dinv.physics.CryoET(
    ctf=False, dose_weighting=False, ice_thickness=0.0, **common
)
ctf_physics = {
    df: dinv.physics.CryoET(defocus=df, dose_weighting=False, **common)
    for df in (2.0, 5.0)
}
dinv.utils.plot(
    [ideal.A(x_gt)[:, :, i0]] + [p.A(x_gt)[:, :, i0] for p in ctf_physics.values()],
    titles=["no CTF"] + [f"defocus {df} µm" for df in ctf_physics],
    suptitle="Projection at 0°",
)

# %%
# The zeros of the CTF produce Thon rings in the power spectrum of projected amorphous ice, close to white noise.

ice = torch.randn(x_gt.shape, generator=generator, device=device)
dinv.utils.plot(
    [
        torch.fft.fftshift(torch.fft.fft2(p.A(ice)[:, :, i0]).abs() ** 2 + 1e-3).log()
        for p in ctf_physics.values()
    ],
    titles=[f"defocus {df} µm" for df in ctf_physics],
    suptitle="Power spectra of projected ice",
)

# %%
# Defocus gradient
# ~~~~~~~~~~~~~~~~
# In tilted or thick specimens, points at different heights along the beam have different defocus.
# This is modelled by `n_defocus_planes` slabs with distinct CTFs, which differ from the thin-specimen approximation more at high tilts.

thin = dinv.physics.CryoET(defocus=3.0, dose_weighting=False, **common)
thick = dinv.physics.CryoET(
    defocus=3.0, dose_weighting=False, n_defocus_planes=8, **common
)
y_thin, y_thick = thin.A(x_gt), thick.A(x_gt)
rel = (y_thick - y_thin).norm(dim=(-1, -2)) / y_thin.norm(dim=(-1, -2))
print(
    f"Relative difference: {rel[0, 0, i0]:.3f} at {angles[i0]:.0f}°, {rel[0, 0, i_lo]:.3f} at {angles[i_lo]:.0f}°"
)
dinv.utils.plot(
    [y_thin[:, :, i_lo], y_thick[:, :, i_lo], y_thick[:, :, i_lo] - y_thin[:, :, i_lo]],
    titles=["single defocus", "8 defocus planes", "difference"],
    suptitle=f"Tilt at {angles[i_lo]:.0f}°",
)

# %%
# Dose, acquisition scheme and ice thickness
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The :func:`acquisition order <deepinv.physics.functional.tilt_acquisition_order>` sets the accumulated exposure of each tilt: the dose-symmetric scheme acquires low tilts first, preserving their high resolution.
# The electron path in the ice grows as :math:`1/\cos\theta`, reducing the intensity at high tilts.

dose_physics = {
    s: dinv.physics.CryoET(defocus=3.0, dose_per_tilt=3.0, tilt_scheme=s, **common)
    for s in ("dose_symmetric", "bidirectional", "unidirectional")
}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
for s, p in dose_physics.items():
    axs[0].plot(ang, p.accumulated_exposure()[0].cpu(), "o", markersize=3, label=s)
for t in (1000.0, 2000.0, 3000.0):
    w = dinv.physics.CryoET(ice_thickness=t, **common).tilt_weights()[0]
    axs[1].plot(ang, w.cpu(), label=f"{t / 10:.0f} nm")
axs[0].set(xlabel="tilt angle (°)", ylabel="e/Å²", title="Accumulated exposure")
axs[1].set(xlabel="tilt angle (°)", title="Relative intensity (ice thickness)")
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()

# %%
# At the highest tilt, acquired last in the unidirectional scheme, high frequencies are strongly attenuated, as seen in the power spectrum perpendicular to the tilt axis.

freqs = torch.fft.rfftfreq(sim_shape[-1], d=PIXEL_SIZE)
plt.figure(figsize=(5, 3.5))
for s in ("dose_symmetric", "unidirectional"):
    tilt = dose_physics[s].A(x_gt)[0, 0, i_hi]
    plt.semilogy(
        freqs, (torch.fft.rfft(tilt, dim=-1).abs() ** 2).mean(0).cpu(), label=s
    )
plt.xlabel("spatial frequency (1/Å)")
plt.title(f"Power spectrum at {angles[i_hi]:.0f}°")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Shot noise
# ~~~~~~~~~~
# With `background=1`, measurements are in units of the mean number of electrons per pixel, and :class:`deepinv.physics.PoissonGaussianNoise` with gain `1 / physics.electrons_per_pixel` models the detector counting statistics.

noisy = dinv.physics.CryoET(defocus=3.0, dose_per_tilt=3.0, background=1.0, **common)
noisy.noise_model = dinv.physics.PoissonGaussianNoise(
    gain=1 / noisy.electrons_per_pixel, sigma=0.01, clip_positive=True, rng=generator
)
print(f"Electrons per pixel and per tilt: {noisy.electrons_per_pixel.item():.0f}")
y_noisy = noisy(x_gt)
x_noisy_fbp = noisy.fbp(noisy.remove_background(y_noisy))

dinv.utils.plot(
    [y_noisy[:, :, i0], y_noisy[:, :, i_lo]],
    titles=[f"{angles[i0]:.0f}°", f"{angles[i_lo]:.0f}°"],
    suptitle="Noisy tilts",
)
dinv.utils.plot_ortho3D(
    [x_gt, noisy.fbp(noisy.A(x_gt)), x_noisy_fbp, F.avg_pool3d(x_noisy_fbp, 4)],
    titles=["ground truth", "WBP (noiseless)", "WBP (noisy)", "WBP (noisy, binned 4x)"],
    figsize=(16, 5),
)

# %%
# Noise dominates the full-resolution WBP, which motivates binning, denoising, and regularized or self-supervised reconstruction, see :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_icecream.py`.

# %%
# Deformations
# ~~~~~~~~~~~~
# Per-tilt deformations of the volume or of the projections model specimen and stage motion. We simulate:
#
# * a misalignment of the projections as a random walk following the acquisition order;
# * a beam-induced deformation of the volume, combining thinning along `z` and a smooth displacement field, both growing with the exposure.
#
# Reconstructing the deformed tilt series with the rigid model blurs the volume.

order = dose_physics["dose_symmetric"].acquisition_order
steps = 0.8 * torch.randn(n_tilts, 2, generator=generator, device=device)
steps[order == 0] = 0.0
drift = torch.cumsum(steps[torch.argsort(order)], dim=0)[order]
progress = order.float() / (n_tilts - 1)
thinning = torch.ones(n_tilts, 3, device=device)
thinning[:, 0] = 1 - 0.1 * progress
field = 3.0 * torch.randn(3, 4, 4, 4, generator=generator, device=device)

rigid = dinv.physics.CryoET(defocus=3.0, **common)
deformed = dinv.physics.CryoET(
    defocus=3.0,
    projection_shift=drift,
    volume_scale=thinning,
    volume_displacement=progress[:, None, None, None, None] * field,
    **common,
)
y_rigid, y_deformed = rigid.A(x_gt), deformed.A(x_gt)
dinv.utils.plot(
    [
        y_rigid[:, :, i_lo],
        y_deformed[:, :, i_lo],
        y_deformed[:, :, i_lo] - y_rigid[:, :, i_lo],
    ],
    titles=["rigid", "deformed", "difference"],
    suptitle=f"Tilt at {angles[i_lo]:.0f}°",
)
dinv.utils.plot_ortho3D(
    [rigid.fbp(y_rigid), rigid.fbp(y_deformed)],
    titles=["WBP of rigid tilt series", "WBP of deformed tilt series"],
    figsize=(8, 5),
)

# %%
# As the operator is differentiable with respect to its parameters, deformations can be estimated jointly with the volume by gradient descent.
#
# :References:
#
# .. footbibliography::
