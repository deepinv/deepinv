r"""
Tour of Ultrafast Ultrasound in DeepInverse
===========================================

Walkthrough of :class:`deepinv.physics.UltrasoundPlaneWave`: forward simulation,
adjoint DAS beamforming, apodization windows, interpolation kernels, transmit
slicing, and speckle phantoms.
"""

import math
import torch
import deepinv as dinv
import matplotlib.pyplot as plt
from scipy.signal import envelope

device = dinv.utils.get_device()
dtype = torch.float32

# %%
# 1. Transducer array and grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 64-element linear array at 0.3 mm pitch, 11 plane waves in ±12°. The 640×192
# grid uses anisotropic spacing (``dz = lambda/5``, ``dx = lambda/2``) so the DAS
# RF image stays above the Nyquist rate.

n_elements = 64
pitch = 3e-4
ele_x = torch.linspace(
    -pitch * (n_elements - 1) / 2, pitch * (n_elements - 1) / 2, n_elements
)
ele_pos = torch.stack([ele_x, torch.zeros(n_elements)], dim=-1).to(device)

angles = torch.linspace(math.radians(-12.0), math.radians(12.0), 21).to(device)
img_size = (640, 192)

print(
    "Transducer active aperture width:", (ele_x.max() - ele_x.min()).item() * 1e3, "mm"
)

# %%
# 2. Target and transducer pulse
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Three point scatterers on a real RF reflectivity of shape ``(B, 1, Z, X)``.
# The pulse-echo impulse response ``h(t)`` is a 1-cycle square-wave excitation convolved
# with a Gaussian pulse (80% fractional bandwidth at 5 MHz).

x = torch.zeros(1, 1, *img_size, device=device)
x[0, 0, 200, 96] = 1.0
x[0, 0, 350, 48] = 1.0
x[0, 0, 500, 144] = 1.0

sound_speed = 1540.0
fc = 5e6
lam = sound_speed / fc
pixel_size = (lam / 5, lam / 2)

sampling_freq = 20e6
fractional_bw = 0.8
sigma_t = math.sqrt(2 * math.log(2)) / (math.pi * fractional_bw * fc)

num_half = math.ceil(3.5 * sigma_t * sampling_freq)
t_pulse = torch.linspace(
    -num_half / sampling_freq, num_half / sampling_freq, 2 * num_half + 1, device=device
)
g_pulse = torch.exp(-(t_pulse**2) / (2 * sigma_t**2)) * torch.cos(
    2 * math.pi * fc * t_pulse
)

num_half_exc = max(1, math.ceil(0.5 * sampling_freq / fc))
exc_signal = torch.cat(
    [torch.ones(num_half_exc, device=device), -torch.ones(num_half_exc, device=device)],
    dim=0,
)
pad_len = exc_signal.numel() // 2
g_padded = torch.nn.functional.pad(g_pulse, (pad_len, pad_len))
pulse = torch.nn.functional.conv1d(
    g_padded.view(1, 1, -1), exc_signal.view(1, 1, -1).flip(-1)
).view(-1)
pulse = pulse / torch.linalg.norm(pulse)

physics = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles,
    element_positions=ele_pos,
    n_samples=1024,
    sampling_frequency=sampling_freq,
    sound_speed=sound_speed,
    pixel_size=pixel_size,
    signal_kind="rf",
    pulse=pulse,
    device=device,
)

# %%
# 3. Forward channel data (y = Ax)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Channel-data shape: ``(B, 1, n_angles, n_elements, n_samples)``.

#y = physics(x)
#
#print("Reflectivity image shape:", x.shape)
#print("Channel-data measurements shape:", y.shape)
#
#envelope_y = torch.from_numpy(
#    envelope(y[0, 0, 2].cpu().numpy(), axis=1, residual=None)
#).to(device)
#bmode_y = 20 * torch.log10(envelope_y / envelope_y.max().clamp(min=1e-12))
#bmode_y = (bmode_y + 40).clamp(min=0.0) / 40
#
#plt.figure(figsize=(6, 4))
#plt.imshow(bmode_y.cpu().numpy(), aspect="auto", cmap="viridis")
#plt.colorbar(label="Amplitude [dB]")
#plt.xlabel("Time Samples")
#plt.ylabel("Receiver Element Index")
#plt.title("Raw Channel Data (Center Transmit, θ = 0°)")
#plt.show()

"""
# %%
# 4. Adjoint DAS beamforming (x = A^T y)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``A_adjoint`` is a coherently-compounded Delay-And-Sum beamformer on real RF data.

x_das = physics.A_adjoint(y)

envelope_das = torch.from_numpy(
    envelope(x_das[0, 0].cpu().numpy(), axis=0, residual=None)
).to(device)
db_das = 20 * torch.log10(envelope_das / envelope_das.max().clamp(min=1e-12))
bmode_das = (db_das + 40).clamp(min=0.0) / 40

dinv.utils.plot(
    [x[0], bmode_das.unsqueeze(0)],
    titles=["Ground Truth Target", "Log-Compressed B-Mode Image"],
    aspect=pixel_size[0] / pixel_size[1],
)

# %%
# 5. Apodization windows
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Receive apodization (``f_number`` + ``rx_apod_window``) sets the acceptance cone;
# transmit apodization (``tx_apod_window``) tapers the aperture. Windows:
# ``"rect"``, ``"hann"``, ``"hamming"``, ``"tukey0.25"``.

physics_apod = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles,
    element_positions=ele_pos,
    n_samples=1024,
    sampling_frequency=sampling_freq,
    sound_speed=sound_speed,
    pixel_size=pixel_size,
    f_number=1.5,
    rx_apod_window="hann",
    tx_apod_window="tukey0.25",
    signal_kind="rf",
    pulse=pulse,
    device=device,
)

x_das_apod = physics_apod.A_adjoint(y)
envelope_das_apod = torch.from_numpy(
    envelope(x_das_apod[0, 0].cpu().numpy(), axis=0, residual=None)
).to(device)
db_das_apod = 20 * torch.log10(
    envelope_das_apod / envelope_das_apod.max().clamp(min=1e-12)
)
bmode_das_apod = (db_das_apod + 40).clamp(min=0.0) / 40

dinv.utils.plot(
    [bmode_das.unsqueeze(0), bmode_das_apod.unsqueeze(0)],
    titles=["Unapodized", "Apodized"],
    aspect=pixel_size[0] / pixel_size[1],
)

# %%
# 6. Interpolation kernels
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Fractional-delay sampling supports ``"nearest"``, ``"linear"`` (default) and
# ``"keys"`` (4-tap Keys cubic convolution).

physics_cubic = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles,
    element_positions=ele_pos,
    n_samples=1024,
    sampling_frequency=sampling_freq,
    sound_speed=sound_speed,
    pixel_size=pixel_size,
    interp="keys",
    signal_kind="rf",
    pulse=pulse,
    device=device,
)

x_das_cubic = physics_cubic.A_adjoint(y)
envelope_das_cubic = torch.from_numpy(
    envelope(x_das_cubic[0, 0].cpu().numpy(), axis=0, residual=None)
).to(device)
db_das_cubic = 20 * torch.log10(
    envelope_das_cubic / envelope_das_cubic.max().clamp(min=1e-12)
)
bmode_das_cubic = (db_das_cubic + 40).clamp(min=0.0) / 40

dinv.utils.plot(
    [bmode_das.unsqueeze(0), bmode_das_cubic.unsqueeze(0)],
    titles=["Linear", "Keys cubic"],
    aspect=pixel_size[0] / pixel_size[1],
)

# %%
# 7. Slicing transmits / compounding
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``select_transmits`` returns a physics operator restricted to a subset of angles —
# useful for self-supervised training on independent sub-apertures.

physics_subset = physics.select_transmits([0, 4])
y_subset = physics_subset(x)

print("Full channel data shape (5 angles):", y.shape)
print("Subset channel data shape (2 angles):", y_subset.shape)
"""

# %%
# 8. Fully-developed speckle (tissue phantom)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Dense random scatterers produce speckle; a circular anechoic cyst is carved out.
# Because ``dz != dx``, the physical circle is an ellipse in pixel coordinates.

torch.manual_seed(42)
background = torch.randn(1, 1, *img_size, device=device) * 0.15

zz, xx = torch.meshgrid(
    torch.arange(img_size[0], device=device),
    torch.arange(img_size[1], device=device),
    indexing="ij",
)
cyst_radius_m = 4.3e-3
rz_pix = cyst_radius_m / pixel_size[0]
rx_pix = cyst_radius_m / pixel_size[1]
cyst_mask = ((xx - 96) / rx_pix) ** 2 + ((zz - 320) / rz_pix) ** 2 <= 1.0
background[:, :, cyst_mask] = 0.0

y_speckle = physics(background)
x_speckle_das = physics.A_adjoint(y_speckle)
envelope_speckle = torch.from_numpy(
    envelope(x_speckle_das[0, 0].cpu().numpy(), axis=0, residual=None)
).to(device)
db_speckle = 20 * torch.log10(
    envelope_speckle / envelope_speckle.max().clamp(min=1e-12)
)
bmode_speckle = (db_speckle + 40).clamp(min=0.0) / 40

#dinv.utils.plot(
#    [torch.abs(background[0]), bmode_speckle.unsqueeze(0)],
#    titles=["Cyst Phantom Target", "B-Mode Speckle Image"],
#    aspect=pixel_size[0] / pixel_size[1],
#)

# %%
# 9. 1 PW DAS vs. CPWC vs. PnP-1PW on the cyst phantom
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Reconstruct the same cyst phantom from three different acquisition/algorithm
# combinations:
#
# * **1 PW DAS**: adjoint beamforming on a single plane wave (center angle).
# * **CPWC**: coherent compounding of all 5 plane waves via the adjoint.
# * **PnP-1PW**: Plug-and-Play reconstruction from the same single plane wave,
#   using Proximal Gradient Descent with a pretrained DRUNet denoiser as prior.
#
# Since RF signals are bipolar and DRUNet is trained on natural images in
# :math:`[0, 1]`, the denoiser is wrapped in a per-sample min/max normalizer
# that rescales the noise level accordingly. The physics operator is normalized
# to unit spectral norm, so ``stepsize=1`` is a safe starting point.

from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import PGD


class MinMaxDenoiser(torch.nn.Module):
    """Wrap a ``[0, 1]``-trained denoiser to handle zero-mean bipolar signals."""

    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x, sigma):
        reduce_dims = tuple(range(1, x.ndim))
        xmin = x.amin(dim=reduce_dims, keepdim=True)
        xmax = x.amax(dim=reduce_dims, keepdim=True)
        span = (xmax - xmin).clamp(min=1e-8)
        x_n = (x - xmin) / span
        sigma_n = float(sigma) / float(span.mean())
        return self.denoiser(x_n, sigma_n) * span + xmin


def _bmode(x_img):
    """Envelope + 40 dB log-compression + normalization for display."""
    env = torch.from_numpy(
        envelope(x_img[0, 0].cpu().numpy(), axis=0, residual=None)
    ).to(x_img.device)
    db = 20 * torch.log10(env / env.max().clamp(min=1e-12))
    return (db + 40).clamp(min=0.0) / 40


# 1 PW acquisition: restrict physics + measurements to the center angle
center_idx = angles.numel() // 2
physics_1pw = physics.select_transmits([center_idx])
y_speckle_1pw = y_speckle[:, :, center_idx : center_idx + 1]

# 1 PW DAS baseline
x_speckle_das_1pw = physics_1pw.A_adjoint(y_speckle_1pw)
bmode_speckle_das_1pw = _bmode(x_speckle_das_1pw)

# PnP from the same single plane wave
denoiser_drunet = dinv.models.DRUNet(
    in_channels=1, out_channels=1, pretrained="download", device=device
)
denoiser_pnp = MinMaxDenoiser(denoiser_drunet).to(device)

# DPIR-style sigma annealing: start with strong denoising (fills the operator
# null space, cleaning the cyst interior), decay to gentle denoising
# (preserves the speckle statistics needed for ultrasound reading).
pnp_max_iter = 30
sigma_schedule = torch.logspace(
    math.log10(3e-2), math.log10(5e-4), pnp_max_iter
)
model_pnp = PGD(
    data_fidelity=L2(),
    prior=PnP(denoiser=denoiser_pnp),
    stepsize=1.0,
    sigma_denoiser=sigma_schedule,
    early_stop=True,
    max_iter=pnp_max_iter,
    verbose=False,
    custom_init=lambda y, physics: physics.A_adjoint(y),
)
model_pnp.eval()

with torch.no_grad():
    x_speckle_pnp_1pw = model_pnp(y_speckle_1pw, physics_1pw)
bmode_speckle_pnp_1pw = _bmode(x_speckle_pnp_1pw)

dinv.utils.plot(
    [torch.abs(background[0]),
     bmode_speckle_das_1pw.unsqueeze(0),
     bmode_speckle.unsqueeze(0),
     bmode_speckle_pnp_1pw.unsqueeze(0)],
    titles=[
        "Cyst Phantom Target",
        "1 PW DAS",
        f"CPWC ({angles.numel()} angles)",
        f"PnP-DRUNet 1 PW (anneal, {pnp_max_iter} iters)",
    ],
    aspect=pixel_size[0] / pixel_size[1],
)

'''
# %%
# 10. EPFL Ultrafast Ultrasound Dataset (volunteer_005)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Load an in-vivo acquisition from the EPFL LTS5 ultrafast ultrasound dataset
# (see https://www.epfl.ch/labs/lts5/research/us/epfl-ultrafast-ultrasound-datasets/).
# The archive is organized as ``volunteer_XXX/<anatomy>/invivo_YYYYY.npz``,
# and acquisition parameters (angles, sampling frequency, probe geometry, …)
# are shared across acquisitions and distributed separately in
# ``settings.zip``. Point ``EPFL_ACQ_PATH`` at a locally extracted ``.npz``
# and fill ``EPFL_SETTINGS`` with the values read from ``settings.zip``.

import sys
from pathlib import Path
import numpy as np

EPFL_ACQ_PATH = Path("/path/to/volunteer_005/abdomen/invivo_14592.npz")

EPFL_SETTINGS = {
    "angles_deg": np.linspace(-16.0, 16.0, 75),  # steering angles [deg]
    "sampling_frequency": 31.25e6,               # [Hz]
    "sound_speed": 1540.0,                       # [m/s]
    "modulation_frequency": None,                # [Hz], None for RF acquisitions
    "n_elements": 128,
    "pitch": 3.0e-4,                             # [m]
    "t0": 0.0,                                   # [s] scalar or (n_angles,) array
    "signal_kind": "rf",                         # "rf" or "iq"
}

if not EPFL_ACQ_PATH.exists():
    print(f"EPFL acquisition not found at {EPFL_ACQ_PATH}; skipping in-vivo sections.")
    sys.exit(0)

_npz = np.load(EPFL_ACQ_PATH)
_data_np = _npz[list(_npz.files)[0]]

if EPFL_SETTINGS["signal_kind"] == "rf":
    y_epfl = torch.as_tensor(_data_np, dtype=dtype, device=device)[None, None]
elif np.iscomplexobj(_data_np):
    y_epfl = torch.stack(
        [torch.as_tensor(_data_np.real, dtype=dtype),
         torch.as_tensor(_data_np.imag, dtype=dtype)],
        dim=0,
    ).to(device).unsqueeze(0)
else:  # (n_angles, n_elements, n_samples, 2) with I/Q on the last axis
    y_epfl = (
        torch.as_tensor(_data_np, dtype=dtype, device=device)
        .moveaxis(-1, 0)
        .unsqueeze(0)
    )

n_a_epfl, n_e_epfl, n_s_epfl = y_epfl.shape[-3:]

ele_x_epfl = torch.linspace(
    -EPFL_SETTINGS["pitch"] * (EPFL_SETTINGS["n_elements"] - 1) / 2,
    EPFL_SETTINGS["pitch"] * (EPFL_SETTINGS["n_elements"] - 1) / 2,
    EPFL_SETTINGS["n_elements"],
)
ele_pos_epfl = torch.stack(
    [ele_x_epfl, torch.zeros_like(ele_x_epfl)], dim=-1
).to(device)

ref_freq_epfl = (
    EPFL_SETTINGS["modulation_frequency"] or EPFL_SETTINGS["sampling_frequency"]
)
lam_epfl = EPFL_SETTINGS["sound_speed"] / ref_freq_epfl
pixel_size_epfl = (lam_epfl / 4.0, lam_epfl / 2.0)

depth_max = (
    EPFL_SETTINGS["sound_speed"] * n_s_epfl / (2.0 * EPFL_SETTINGS["sampling_frequency"])
)
img_size_epfl = (int(round(depth_max / pixel_size_epfl[0])), 256)

angles_epfl = torch.deg2rad(
    torch.as_tensor(EPFL_SETTINGS["angles_deg"], dtype=dtype)
).to(device)

physics_epfl = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size_epfl,
    angles=angles_epfl,
    element_positions=ele_pos_epfl,
    n_samples=n_s_epfl,
    sampling_frequency=EPFL_SETTINGS["sampling_frequency"],
    sound_speed=EPFL_SETTINGS["sound_speed"],
    pixel_size=pixel_size_epfl,
    demodulation_frequency=EPFL_SETTINGS["modulation_frequency"],
    t0=EPFL_SETTINGS["t0"],
    signal_kind=EPFL_SETTINGS["signal_kind"],
    f_number=1.5,
    rx_apod_window="hann",
    normalize=True,
    device=device,
    dtype=dtype,
)

print("EPFL channel data shape:", tuple(y_epfl.shape))
print("EPFL image grid (Z, X):", img_size_epfl)
print(f"Imaging depth: {depth_max * 1e3:.1f} mm")

# %%
# 11. Single Plane Wave vs. Coherent Plane-Wave Compounding (CPWC)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Both reconstructions are adjoint operators. 1 PW uses only the center
# steering angle (via :meth:`select_transmits`); CPWC uses all angles — the
# adjoint sums coherently across transmits by construction.

center_idx = n_a_epfl // 2
physics_1pw = physics_epfl.select_transmits([center_idx])
y_1pw = y_epfl[:, :, center_idx : center_idx + 1]
x_1pw = physics_1pw.A_adjoint(y_1pw)
x_cpwc = physics_epfl.A_adjoint(y_epfl)


def _bmode(x_img):
    """Envelope + 40 dB log-compression + normalization for display."""
    ch = x_img[0]
    if ch.shape[0] == 2:
        env = torch.hypot(ch[0], ch[1])
    else:
        env = torch.from_numpy(
            envelope(ch[0].cpu().numpy(), axis=0, residual=None)
        ).to(ch.device)
    db = 20 * torch.log10(env / env.max().clamp(min=1e-12))
    return (db + 40).clamp(min=0.0) / 40


bmode_1pw = _bmode(x_1pw).unsqueeze(0)
bmode_cpwc = _bmode(x_cpwc).unsqueeze(0)

# %%
# 12. Plug-and-Play reconstruction on real data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Reuse the :class:`MinMaxDenoiser` / :class:`PGD` setup from section 9. In IQ
# mode the pretrained DRUNet is instantiated with two input channels; for RF
# acquisitions the same 1-channel denoiser as the cyst-phantom example applies.

n_chan_epfl = 1 if EPFL_SETTINGS["signal_kind"] == "rf" else 2
if n_chan_epfl != 1:
    denoiser_drunet = dinv.models.DRUNet(
        in_channels=n_chan_epfl,
        out_channels=n_chan_epfl,
        pretrained="download",
        device=device,
    )
    denoiser_pnp = MinMaxDenoiser(denoiser_drunet).to(device)
    model_pnp = PGD(
        data_fidelity=L2(),
        prior=PnP(denoiser=denoiser_pnp),
        stepsize=1.0,
        sigma_denoiser=0.03,
        early_stop=True,
        max_iter=pnp_max_iter,
        verbose=False,
        custom_init=lambda y, physics: physics.A_adjoint(y),
    )
    model_pnp.eval()

with torch.no_grad():
    x_pnp = model_pnp(y_epfl, physics_epfl)

bmode_pnp = _bmode(x_pnp).unsqueeze(0)

# %%
# 13. Side-by-side comparison
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# CPWC sharpens the PSF and reduces sidelobes compared to a single plane wave;
# PnP additionally suppresses out-of-support noise thanks to the learned prior.

dinv.utils.plot(
    [bmode_1pw, bmode_cpwc, bmode_pnp],
    titles=[
        f"1 PW (angle idx {center_idx})",
        f"CPWC ({n_a_epfl} angles)",
        f"PnP-DRUNet ({pnp_max_iter} iters)",
    ],
    aspect=pixel_size_epfl[0] / pixel_size_epfl[1],
)
'''

# %%
# :References:
#
# .. footbibliography::
