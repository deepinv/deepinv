r"""
Plug-and-Play for in-vivo ultrafast ultrasound
==============================================

This example shows how to use :class:`deepinv.physics.UltrasoundPlaneWave` to reconstructs an in-vivo carotid acquisition of the
`EPFL LTS5 ultrafast ultrasound dataset <https://www.epfl.ch/labs/lts5/research/us/epfl-ultrafast-ultrasound-datasets/>`__.
It compares, on the very same per-channel raw data:

1. **Low quality DAS (single-plane wave imaging)**: the adjoint of the operator restricted to 1 plane-wave transmit. Fast but poor quality due to high amount of sidelobes and low signal to noise ratio.
2. **High quality DAS (CPWC)**: the adjoint of the full operator, i.e. coherent plane-wave compounding
   of all transmitted angles. Our reference image quality, at the cost of one
   acquisition per angle.
3. **Least squares**: :func:`A_dagger <deepinv.physics.LinearPhysics.A_dagger>` on
   the single plane wave acquisition, solved by conjugate gradient.
4. **Plug-and-Play**: proximal gradient descent
   (:class:`deepinv.optim.optimizers.PGD`) on the single-plane wave acquisition with a
   :class:`PnP <deepinv.optim.prior.PnP>` prior, using wavelet and BM3D denoisers.

The raw acquisition is fetched from the DeepInverse HuggingFace repository and
cached locally, so no manual download is required.
"""

import math

import numpy as np
import torch

import deepinv as dinv

# %%
# 1. Configuration
# ----------------
#
# The data were acquired with a GE 9L-D linear array (192 elements, 0.23 mm pitch, 5.3 MHz center frequency) on a Verasonics scanner.
# Per-channel raw RF data were sampled at 20.8 MHz, and 87 plane waves steered between -16.3 and +16.3 degrees were used in transmit. The values below are those of the
# dataset, taken from its acquisition metadata, as the archive downloaded in section 3
# only contains the raw data.
#
# To run it on your own acquisition, replace:
#
# - the imaged region: ``z_min``, ``z_max`` and ``x_half`` in meters, chosen here to
#   cover the carotid artery, which lies a couple of centimeters deep.
# - the probe, in section 2: ``center_freq`` and ``frac_bw``, the pulse-echo fractional
#   bandwidth;
#   ``sampling_freq``; and ``element_positions`` in meters;
# - the sequence, in section 2: ``angles`` in radians and ``t0``, the time of the first recorded sample relative to
#   the plane wave crossing the center of the array.
# - the data, in section 3: real radio-frequency per-channel rawdata of shape
#   ``(n_angles, n_elements, n_samples)``. Note that IQ data are not supported;
# - the speed of sound, passed to the operator in section 4, if 1540 m/s does not suit
#   your medium.
#
# The reconstruction settings, i.e. f-number, apodization, iterations and denoising
# levels, are set where they are used and are worth revisiting for a different probe.
z_min, z_max = 2e-3, 45e-3
x_half = 18e-3

device = dinv.utils.get_device()

# %%
# 2. Acquisition Settings
# -----------------------
#
# The probe geometry, the sampling settings and the transmit sequence of the dataset.
# The elements are laid out along the lateral axis at a constant pitch and centered on
# zero, which fixes the origin of the lateral coordinate of the pixel grid.
center_freq = 5.3e6
frac_bw = 0.75 / np.sqrt(2)
n_elements = 192
pitch = 2.3e-4
element_width = 6.1206151719371e-05
sampling_freq = 20833333.333333332

ele_x = torch.arange(n_elements, dtype=torch.float32) * pitch + element_width / 2
ele_x = ele_x - ele_x.mean()
element_positions = torch.stack([ele_x, torch.zeros_like(ele_x)], dim=-1).to(device)
angles = math.radians(0.38) * torch.tensor(
    [k * sign for k in range(43, 0, -1) for sign in (-1, 1)] + [0]
)

# Initial time to apply to the data for beamforming: t0 = -time_axis[0] + (peak_time - lens_correction)
t0 = 4.272e-06 + (4.1e-07 - 1.92e-07)

# %%
# 3. RF Per-channel Raw Data
# --------------------------
#
# The dataset provides radio-frequency (RF) channel data of shape
# ``(n_angles, n_elements, n_samples)`` which we feed directly to the operator:
# :class:`UltrasoundPlaneWave <deepinv.physics.UltrasoundPlaneWave>` works on real RF
# signals ``y`` of shape ``(B, 1, n_angles, n_elements, n_samples)``.
npz = np.load(
    dinv.utils.load_url(
        "https://huggingface.co/datasets/deepinv/images/resolve/main/"
        "epfl_ufus_carotid_invivo_16654.npz"
    )
)
rf = npz["data"][0].astype(np.float32)
rf = rf / np.abs(rf).max()
y = torch.as_tensor(rf)[None, None].to(device)
del rf, npz

# %%
# 4. The High-quality Forward Operator
# -----------------------------------
#
# We instantiate :class:`deepinv.physics.UltrasoundPlaneWave` with all 87 angles (used for coherent plane-wave
# compounding). This will be our reference.
lam = 1540.0 / center_freq
pixel_size = (lam / 8, pitch / 1.5)
img_size = (
    int((z_max - z_min) / pixel_size[0]),
    int(2 * x_half / pixel_size[1]),
)
pixel_origin = (z_min, -x_half)
pulse_bandwidth_hz = frac_bw * center_freq
sigma_t = math.sqrt(2 * math.log(2)) / (math.pi * pulse_bandwidth_hz)
n_half = math.ceil(3.5 * sigma_t * sampling_freq)
t_pulse = (np.arange(2 * n_half + 1) - n_half) / sampling_freq
pulse_rf = np.exp(-(t_pulse**2) / (2 * sigma_t**2)) * np.cos(
    2 * np.pi * center_freq * t_pulse
)
pulse = torch.as_tensor(pulse_rf.copy(), dtype=torch.float32)

physics = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles,
    element_positions=element_positions,
    n_samples=y.shape[-1],
    sampling_frequency=sampling_freq,
    sound_speed=1540.0,
    pixel_size=pixel_size,
    pixel_origin=pixel_origin,
    t0=t0,
    pulse=pulse,
    f_number=1.5,
    receive_apod_window="hann",
    normalize=False,
    device=device,
)

x_cpwc = physics.A_adjoint(y)

# %%
# 5. The Low-quality Forward Operator
# -----------------------------------
#
# We build a second operator restricted to the single plane-wave transmit with normal incidence (0 degrees).
# In this case, the problem is severely ill-conditioned as the number of projections is restricted comapred to the reference.

fast_idx = [int(angles.abs().argmin())]

physics_fast = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles[fast_idx],
    element_positions=element_positions,
    n_samples=y.shape[-1],
    sampling_frequency=sampling_freq,
    sound_speed=1540.0,
    pixel_size=pixel_size,
    pixel_origin=pixel_origin,
    t0=t0,
    pulse=pulse,
    f_number=1.5,
    receive_apod_window="hann",
    normalize=False,
    device=device,
)
y_fast = y[:, :, fast_idx]

# %%
# 6. Setting the Baseline: Delay-and-sum beamforming
# --------------------------------------------------
# As a baseline, we reconstruct the image using the adjoint operator i.e. the so-called delay-and-sum (DAS) beamforming.
x_fast = physics_fast.A_adjoint(y_fast)

# %%
# 7. Going one step-further: Least-squares reconstruction
# -------------------------------------------------------
#
# As a first try to improve the image quality compared to the baseline, we solve the least-squares (LS) problem :math:`\min_x \|Ax - y\|^2` for the single-plane wave imaging experiment by applying
# the conjugate gradient algorithm. As the problem is severly ill-conditioned the LS estimate is of relatively bad quality (see Section 9).
x_pinv = physics_fast.A_dagger(y_fast, solver="CG", max_iter=20, tol=1e-10)

# %%
# 8. Plug-and-Play Reconstruction
# -------------------------------
#
# In order to overcome the ill-coniditioning of the forward operator, one may inject some a priori knowledge on the RF data in the inverse problem. These priors can be explicit e.g. sparsity in some basis, or more elaborate e.g. lying in the fixed point set of some generic denoisers.
# This leads to the well-known plug and play reconstruction which relies on the proximal gradient descent algorithm (see :class:`deepinv.optim.PGD`) along with the plug-and-play prior (see :class:`deepinv.optim.PnP`)
data_fidelity = dinv.optim.L2()
lipschitz_fast = physics_fast.compute_norm(
    torch.randn(1, 1, *img_size, device=device, dtype=torch.float32),
    max_iter=100,
    tol=1e-4,
    verbose=False,
)
step_size = 1.99 / lipschitz_fast.item()
image_scale = x_pinv.std().item()
x_init = x_fast * (image_scale / x_fast.std())

# %%
# 8.a Wavelet prior
# ^^^^^^^^^^^^^^^^^
#
# As a first prior, we rely on sparsity in the wavelet basis, i.e. we solve
# :math:`\min_x \tfrac{1}{2}\|Ax - y\|^2 + \lambda \|\Psi x\|_1` where :math:`\Psi` is
# an orthonormal wavelet transform. The proximity operator of :math:`\|\Psi \cdot\|_1`
# is the soft-thresholding wavelet denoiser and PGD with
# :class:`deepinv.optim.WaveletPrior` amounts to iterative soft-thresholding — the
# baseline for ultrafast ultrasound imaging.
print("\nPGD with the wavelet sparsity prior")

prior_wavelet = dinv.optim.WaveletPrior(level=3, wv="db4", p=1, device=device)

lambda_reg_wavelet = 0.5

model_wavelet = dinv.optim.PGD(
    data_fidelity=data_fidelity,
    prior=prior_wavelet,
    stepsize=step_size,
    lambda_reg=lambda_reg_wavelet,
    max_iter=50,
    early_stop=True,
    verbose=True,
    show_progress_bar=True,
    custom_init=lambda y, physics: {"est": (x_init,)},
)
model_wavelet.eval()
with torch.no_grad():
    x_pnp_wavelet, metrics_wavelet = model_wavelet(
        y_fast, physics_fast, compute_metrics=True
    )
dinv.utils.plot_curves({"residual": metrics_wavelet["residual"]})

# %%
# 8.b BM3D prior
# ^^^^^^^^^^^^^^
#
# As a second prior, we rely on a plug-and-play approximation (:class:`deepinv.optim.PnP`) i.e. the image lies in the
# fixed point set of the generic BM3D denoiser (:class:`deepinv.models.BM3D`).
print("\nPnP with the BM3D prior")

pnp_max_iter = 20
sigma_denoiser = image_scale * torch.logspace(
    math.log10(0.15), math.log10(0.03), pnp_max_iter
)
denoiser_bm3d = dinv.models.BM3D(device=device)

model_bm3d = dinv.optim.PGD(
    data_fidelity=data_fidelity,
    prior=dinv.optim.PnP(denoiser=denoiser_bm3d),
    stepsize=step_size,
    sigma_denoiser=sigma_denoiser,
    max_iter=pnp_max_iter,
    early_stop=True,
    verbose=True,
    show_progress_bar=True,
    custom_init=lambda y, physics: {"est": (x_init,)},
)
model_bm3d.eval()
with torch.no_grad():
    x_pnp_bm3d, metrics_bm3d = model_bm3d(y_fast, physics_fast, compute_metrics=True)

dinv.utils.plot_curves({"residual": metrics_bm3d["residual"]})

# %%
# B-Mode images
# -------------
#
# Each reconstruction is converted to a B-mode image with :func:`deepinv.utils.bmode`:
# envelope of the RF signal along depth, normalized by its maximum and log-compressed
# over ``dynamic_range`` dB.:

dynamic_range = 50.0

bmodes = {
    "1 PW DAS": dinv.utils.bmode(x_fast, dim=-2, amplitude_floor_db=-dynamic_range),
    f"CPWC ({angles.numel()} angles)": dinv.utils.bmode(
        x_cpwc, dim=-2, amplitude_floor_db=-dynamic_range
    ),
    "Least squares 1 PW": dinv.utils.bmode(
        x_pinv, dim=-2, amplitude_floor_db=-dynamic_range
    ),
    "Wavelet-sparsity 1 PW": dinv.utils.bmode(
        x_pnp_wavelet, dim=-2, amplitude_floor_db=-dynamic_range
    ),
    "PnP-BM3D 1 PW": dinv.utils.bmode(
        x_pnp_bm3d, dim=-2, amplitude_floor_db=-dynamic_range
    ),
}

extent = [-x_half * 1e3, x_half * 1e3, z_max * 1e3, z_min * 1e3]
dinv.utils.plot(
    bmodes,
    rescale_mode="clip",
    vmin=-dynamic_range,
    vmax=0.0,
    cmap="gray",
    extent=extent,
    aspect="equal",
    figsize=(3 * len(bmodes), 5),
    suptitle=f"B-mode, {dynamic_range:.0f} dB dynamic range",
)
