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
# 64-element linear array at 0.3 mm pitch, 5 plane waves in ±12°. The 1024×192
# grid uses anisotropic spacing (``dz = lambda/8``, ``dx = lambda/2``) so the DAS
# RF image stays above the Nyquist rate.

n_elements = 64
pitch = 3e-4
ele_x = torch.linspace(
    -pitch * (n_elements - 1) / 2, pitch * (n_elements - 1) / 2, n_elements
)
ele_pos = torch.stack([ele_x, torch.zeros(n_elements)], dim=-1).to(device)

angles = torch.linspace(math.radians(-12.0), math.radians(12.0), 5).to(device)
img_size = (1024, 192)

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
x[0, 0, 320, 96] = 1.0
x[0, 0, 560, 48] = 1.0
x[0, 0, 800, 144] = 1.0

sound_speed = 1540.0
fc = 5e6
lam = sound_speed / fc
pixel_size = (lam / 8, lam / 2)

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

y = physics(x)

print("Reflectivity image shape:", x.shape)
print("Channel-data measurements shape:", y.shape)

envelope_y = torch.from_numpy(
    envelope(y[0, 0, 2].cpu().numpy(), axis=1, residual=None)
).to(device)
bmode_y = 20 * torch.log10(envelope_y / envelope_y.max().clamp(min=1e-12))
bmode_y = (bmode_y + 40).clamp(min=0.0) / 40

plt.figure(figsize=(6, 4))
plt.imshow(bmode_y.cpu().numpy(), aspect="auto", cmap="viridis")
plt.colorbar(label="Amplitude [dB]")
plt.xlabel("Time Samples")
plt.ylabel("Receiver Element Index")
plt.title("Raw Channel Data (Center Transmit, θ = 0°)")
plt.show()

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
cyst_mask = ((xx - 96) / rx_pix) ** 2 + ((zz - 512) / rz_pix) ** 2 <= 1.0
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

dinv.utils.plot(
    [torch.abs(background[0]), bmode_speckle.unsqueeze(0)],
    titles=["Cyst Phantom Target", "B-Mode Speckle Image"],
    aspect=pixel_size[0] / pixel_size[1],
)

# %%
# :References:
#
# .. footbibliography::
