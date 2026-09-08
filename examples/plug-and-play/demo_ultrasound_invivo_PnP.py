r"""
Plug-and-Play for in-vivo ultrafast ultrasound
==============================================

This example shows how to use :class:`deepinv.physics.UltrasoundPlaneWave` to reconstructs an in-vivo carotid acquisition of the
`EPFL LTS5 ultrafast ultrasound dataset <https://www.epfl.ch/labs/lts5/research/us/epfl-ultrafast-ultrasound-datasets/>`__.
It compares, on the very same per-channel raw data:

1. **Low quality DAS**: the adjoint of the operator restricted to 1 plane-wave transmit. Fast but poor quality due to high amount of sidelobes and low signal to noise ratio.
2. **High quality DAS (CPWC)**: the adjoint of the full operator, i.e. coherent plane-wave compounding
   of all transmitted angles. Our reference image quality, at the cost of one
   acquisition per angle.
3. **Least squares**: :func:`A_dagger <deepinv.physics.LinearPhysics.A_dagger>` on
   the short sequence, solved by conjugate gradient.
4. **Plug-and-Play**: proximal gradient descent
   (:class:`deepinv.optim.optimizers.PGD`) on the short sequence with a
   :class:`PnP <deepinv.optim.prior.PnP>` prior, using wavelet and BM3D denoisers.

The raw acquisition is fetched from the DeepInverse HuggingFace repository and
cached locally, so no manual download is required.
"""

import math

import numpy as np
import torch
from scipy.signal import envelope
import matplotlib.pyplot as plt

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import PGD
from deepinv.optim.prior import PnP
from deepinv.utils import load_url

# %%
# 1. Configuration
# ----------------
#
# The imaged region (in meters) and the reconstruction settings.
Z_MIN, Z_MAX = 2e-3, 45e-3
X_HALF = 18e-3

PINV_MAX_ITER = 20

F_NUMBER = 1.5
RX_WINDOW = "hann"
PNP_MAX_ITER = 50
SIGMA_START, SIGMA_END = 0.15, 0.03

device = dinv.utils.get_device()
dtype = torch.float32

# %%
# 2. Acquisition Settings
# -----------------------
#
# The probe geometry, the transmitted steering angles and the sampling settings
# for the GE 9L-D linear array used in the EPFL LTS5 ultrafast ultrasound
# dataset.

sound_speed = 1540.0
center_freq = 5.3e6
# Pulse-echo -6 dB fractional bandwidth (transmit-times-receive → 0.75 / sqrt(2)).
frac_bw = 0.75 / np.sqrt(2)
n_elements = 192
pitch = 2.3e-4
element_width = 6.1206151719371e-05
sampling_freq = 20833333.333333332

ele_x = torch.arange(n_elements, dtype=dtype) * pitch + element_width / 2
ele_x = ele_x - ele_x.mean()
element_positions = torch.stack([ele_x, torch.zeros_like(ele_x)],
                                dim=-1).to(device)

n_angles = 87
spacing_rad = math.radians(0.38)
_half = (n_angles - 1) // 2
_mags = torch.arange(_half, 0, -1, dtype=dtype) * spacing_rad
angles = torch.zeros(n_angles, dtype=dtype)
angles[0:2 * _half:2] = -_mags
angles[1:2 * _half:2] = _mags

# t0 = -time_axis[0] + (peak_time - lens_correction)
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
    load_url("https://huggingface.co/datasets/deepinv/images/resolve/main/"
             "epfl_ufus_carotid_invivo_16654.npz"))
rf = npz["data"][0].astype(np.float32)
rf = rf / np.abs(rf).max()
n_samples = rf.shape[-1]
y = torch.as_tensor(rf, dtype=dtype)[None, None].to(device)
del rf

# %%
# 4. The High-quality Forward Operator
# -----------------------------------
#
# First, we instantiate :class:`deepinv.physics.UltrasoundPlaneWave` with all 87 angles (used for coherent plane-wave
# compounding). This will be our reference.
lam = sound_speed / center_freq
pixel_size = (lam / 6, pitch / 1.5)
img_size = (
    int((Z_MAX - Z_MIN) / pixel_size[0]),
    int(2 * X_HALF / pixel_size[1]),
)
pixel_origin = (Z_MIN, -X_HALF)

# Pulse-echo impulse response: a Gaussian-modulated cosine at the probe carrier.
pulse_bandwidth_hz = frac_bw * center_freq
sigma_t = math.sqrt(2 * math.log(2)) / (math.pi * pulse_bandwidth_hz)
n_half = math.ceil(3.5 * sigma_t * sampling_freq)
t_pulse = (np.arange(2 * n_half + 1) - n_half) / sampling_freq
pulse_rf = np.exp(-(t_pulse**2) /
                  (2 * sigma_t**2)) * np.cos(2 * np.pi * center_freq * t_pulse)
pulse = torch.as_tensor(pulse_rf.copy(), dtype=dtype)

physics = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles,
    element_positions=element_positions,
    n_samples=n_samples,
    sampling_frequency=sampling_freq,
    sound_speed=sound_speed,
    pixel_size=pixel_size,
    pixel_origin=pixel_origin,
    t0=t0,
    pulse=pulse,
    f_number=F_NUMBER,
    receive_apod_window=RX_WINDOW,
    normalize=False,
    device=device,
)

# %%
# 5. The Low-quality Forward Operator
# -----------------------------------
#
# We build a second operator restricted to the single plane-wave transmit closest
# to broadside (0 degrees). In this case, the problem is severely ill-posed as the number of projections is restricted.

fast_idx = [int(angles.abs().argmin())]

physics_fast = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles[fast_idx],
    element_positions=element_positions,
    n_samples=n_samples,
    sampling_frequency=sampling_freq,
    sound_speed=sound_speed,
    pixel_size=pixel_size,
    pixel_origin=pixel_origin,
    t0=t0,
    pulse=pulse,
    f_number=F_NUMBER,
    receive_apod_window=RX_WINDOW,
    normalize=False,
    device=device,
)
y_fast = y[:, :, fast_idx]

# %%
# 6. Setting the Baselines: DAS and Least-squares Reconstructions
# ---------------------------------------------------------------
#
# The adjoint of the forward operator is the well-known DAS beamforming with CPWC in case of multiple transmits. We also solve the
# least-squares problem :math:`\min_x \|Ax - y\|^2` on the short sequence by
# conjugate gradient.
#
# ``A_adjoint(y)`` scales as :math:`\|A\|^2 x`, and :math:`\|A\|^2` grows with the
# number of transmits, so CPWC would be intrinsically ~N times brighter than the
# single-PW DAS. We divide each adjoint by its own squared spectral norm to remove
# that bias. The least-squares solution needs no such rescaling.

lipschitz_fast = physics_fast.compute_norm(
    torch.randn(1, 1, *img_size, device=device, dtype=dtype),
    max_iter=20,
    tol=1e-4,
    verbose=False,
)
lipschitz_full = physics.compute_norm(
    torch.randn(1, 1, *img_size, device=device, dtype=dtype),
    max_iter=20,
    tol=1e-4,
    verbose=False,
)

x_fast = physics_fast.A_adjoint(y_fast) / lipschitz_fast
x_cpwc = physics.A_adjoint(y) / lipschitz_full
x_pinv = physics_fast.A_dagger(y_fast,
                               solver="CG",
                               max_iter=PINV_MAX_ITER,
                               tol=1e-10)

# %%
# 7. Plug-and-Play Reconstruction
# -------------------------------
#
# We now solve the inverse problem involving the low-quality operator with PGD and a PnP prior. Off-the-shelf
# denoisers expect natural images in :math:`[0, 1]`; the wrapper below standardizes
# the RF image by its standard deviation, centers it and scales the noise level
# accordingly, so that the denoiser sees a signal it was trained for.

denoisers = {
    "wavelet":
    dinv.models.WaveletDenoiser(level=3,
                                wv="db4",
                                non_linearity="soft",
                                device=device),
    "BM3D":
    dinv.models.BM3D(device=device),
}

data_fidelity = L2()
step_size = 1.9 / lipschitz_fast.item()
sigma_denoising_schedule = torch.logspace(math.log10(SIGMA_START),
                                          math.log10(SIGMA_END), PNP_MAX_ITER)

# The PnP prior wraps each grayscale denoiser: the RF image is standardized to
# the unit range the denoiser expects (K std-devs mapped onto [0, 1]) and rescaled
# back afterwards. The data-fit metric is the relative residual computed after
# fitting the global amplitude that a PnP reconstruction is only defined up to.
K = 4.0


def data_fit_metric(history, x_prev, x):
    """Relative residual after fitting the global amplitude scale."""
    ax = physics_fast.A(x.unsqueeze(0))
    amplitude = (ax * y_fast).sum() / ax.pow(2).sum()
    return ((amplitude * ax - y_fast).norm() / y_fast.norm()).item()


x_pnp = {}
for name, denoiser in denoisers.items():
    print(f"\nPnP with the {name} prior")

    def wrap_denoiser(x, sigma, d=denoiser):
        """Standardize x so K std-devs map to unit range around 0.5, denoise, rescale back."""
        scale = K * x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
        denoised = d(x / scale + 0.5, float(sigma) / K)
        return (denoised - 0.5) * scale

    model = PGD(
        data_fidelity=data_fidelity,
        prior=PnP(denoiser=wrap_denoiser),
        stepsize=step_size,
        sigma_denoiser=sigma_denoising_schedule,
        max_iter=PNP_MAX_ITER,
        early_stop=True,
        verbose=True,
        show_progress_bar=True,
        custom_metrics={"data_fit": data_fit_metric},
        custom_init=lambda y, physics: {"est": (x_fast, )},
    )
    model.eval()
    with torch.no_grad():
        x_pnp[name], metrics = model(y_fast,
                                     physics_fast,
                                     compute_metrics=True)

    residual, data_fit = metrics["residual"][0], metrics["data_fit"][0]
    print(f"{'iter':>6s}{'residual':>15s}{'data_fit':>15s}")
    for it in list(range(0, len(residual), 6)) + [len(residual) - 1]:
        print(f"{it:6d}{residual[it]:15.4g}{data_fit[it]:15.4g}")

# %%
# 8. B-mode images
# ----------------

DYNAMIC_RANGE = 50.0

recons = [
    (x_fast, "1 PW DAS"),
    (x_cpwc, f"CPWC ({angles.numel()} angles)"),
    (x_pinv, f"Least squares 1 PW (CG {PINV_MAX_ITER})"),
] + [(v, f"PnP-{k} 1 PW") for k, v in x_pnp.items()]

# Envelope of the bandpass RF signal along depth.
envelopes = [
    envelope(x[0, 0].cpu().numpy(), axis=0, residual=None) for x, _ in recons
]
envelopes = [e / e.max() for e in envelopes]
display_ref = 1.0

extent = [-X_HALF * 1e3, X_HALF * 1e3, Z_MAX * 1e3, Z_MIN * 1e3]
fig, axes = plt.subplots(1,
                         len(recons),
                         figsize=(4.3 * len(recons), 6),
                         constrained_layout=True)
for ax, env, (_, title) in zip(axes, envelopes, recons):
    bmode = 20 * np.log10(env / display_ref + 1e-20)
    ax.imshow(bmode,
              cmap="gray",
              vmin=-DYNAMIC_RANGE,
              vmax=0,
              extent=extent,
              aspect="equal")
    ax.set_title(f"{title}\n(peak {bmode.max():.1f} dB)", fontsize=10)
    ax.set_xlabel("x [mm]")
axes[0].set_ylabel("z [mm]")
plt.show()
