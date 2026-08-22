r"""
Tour of Ultrafast Ultrasound functionality in DeepInverse
=========================================================

This example presents the various forward physics, apodization windows,
and interpolation kernels available in DeepInverse for Ultrafast Ultrasound imaging:

-  Physics: :class:`deepinv.physics.UltrasoundPlaneWave` (and base class :class:`deepinv.physics.UltrafastUltrasound`)
-  Apodization: dynamic receive f-number cone and transmit plane-wave aperture windowing
-  Interpolation: Nearest-neighbor, Bilinear, and Keys' cubic convolution kernels

Contents:

1. Setup Transducer Array and Grid
2. Create Target and Transducer Pulse
3. Simulate Forward Channel Data (y = Ax)
4. Reconstruct via Adjoint DAS Beamforming (x = A^T y)
5. Explore Apodization Windows
6. Explore Interpolation Kernels
7. Slicing Transmits / Compounding
8. Simulate Fully-Developed Speckle (Tissue Phantom)

"""

# %%
import sys
import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, basedir)

import math
import torch
import deepinv as dinv
import matplotlib.pyplot as plt

device = dinv.utils.get_device()
dtype = torch.float32


def hilbert_envelope(rf_tensor):
    """Compute the analytical envelope of a 2D RF tensor along the Z-axis (axis 0) using Hilbert transform."""
    n = rf_tensor.shape[0]
    f = torch.fft.fft(rf_tensor, dim=0)

    # Create the Hilbert mask along the first dimension (Z-axis)
    mask = torch.zeros(n, 1, device=rf_tensor.device)
    mask[0] = 1.0
    if n % 2 == 0:
        mask[n // 2] = 1.0
        mask[1 : n // 2] = 2.0
    else:
        mask[1 : (n + 1) // 2] = 2.0

    analytic = torch.fft.ifft(f * mask, dim=0)
    return torch.abs(analytic)


# %%
# 1. Setup Transducer Array and Grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First, we define a standard linear transducer array. We set up:
#
# - A 64-element array with 0.3 mm pitch (element spacing).
# - 5 transmit plane-wave steering angles spread symmetrically between -12 and +12 degrees.
# - A 1024x192 pixel grid with anisotropic spacing ``dz = lambda/8`` axial and
#   ``dx = lambda/2`` lateral. The fine axial spacing is required so the DAS RF
#   image is sampled well above the round-trip carrier Nyquist rate
#   ``c/(4 f_c) = lambda/4``; otherwise, envelope detection along Z produces
#   strong axial oscillations at pixel scale.
#

n_elements = 64
pitch = 3e-4  # 0.3 mm pitch
ele_x = torch.linspace(
    -pitch * (n_elements - 1) / 2, pitch * (n_elements - 1) / 2, n_elements
)
ele_pos = torch.stack([ele_x, torch.zeros(n_elements)], dim=-1).to(device)

angles = torch.linspace(math.radians(-12.0), math.radians(12.0), 5).to(device)

# Spatial grid (Z, X). Z is 4x larger so we can afford dz = lambda/8 while
# keeping the same ~40 mm axial imaging depth.
img_size = (1024, 192)

print(
    "Transducer active aperture width:", (ele_x.max() - ele_x.min()).item() * 1e3, "mm"
)

# %%
# 2. Create Target and Transducer Pulse
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we simulate a target in our tissue containing a few sub-wavelength **point scatterers**
# (represented as delta functions) placed at different depths.
#
# Since we work directly with raw, un-demodulated **Radio-Frequency (RF)** signals (which is
# standard for direct RF beamforming), the image has a single channel representing the real-valued
# reflectivity, hence shape ``(B, 1, Z, X)``.
#

x = torch.zeros(1, 1, *img_size, device=device)

# Place 3 point scatterers in our high-resolution grid. In physical coordinates
# the pixel positions below correspond to depths of ~12.3, 21.6, and 30.8 mm.
x[0, 0, 320, 96] = 1.0  # Shallow, center
x[0, 0, 560, 48] = 1.0  # Mid-depth, left
x[0, 0, 800, 144] = 1.0  # Deep, right

# We use the standard speed of sound in soft tissue (1540 m/s) and a 5 MHz probe
sound_speed = 1540.0
fc = 5e6

# Anisotropic pixel spacing (see section 1). At c = 1540 m/s and fc = 5 MHz,
# lambda = 308 um, so dz = lambda/8 = 38.5 um and dx = lambda/2 = 154 um.
lam = sound_speed / fc
pixel_size = (lam / 8, lam / 2)

# A physical ultrasound transducer has a bandpass **pulse-echo impulse response** :math:`h(t)`.
# Following clinical models, we can generate this response as the convolution of:
#
# 1. An **excitation signal**: a 1-cycle square wave at the probe's center frequency (5 MHz).
# 2. A **Gaussian pulse**: a bandpass pulse with an 80% fractional bandwidth (at -6 dB) centered at 5 MHz.
#

sampling_freq = 20e6

# Calculate sigma_t for the Gaussian envelope corresponding to 80% fractional bandwidth
fractional_bw = 0.8
sigma_t = math.sqrt(2 * math.log(2)) / (math.pi * fractional_bw * fc)

# Time vector for the Gaussian pulse (centered at 0, spanning +/- 3.5 sigma)
num_half = math.ceil(3.5 * sigma_t * sampling_freq)
t_pulse = torch.linspace(
    -num_half / sampling_freq, num_half / sampling_freq, 2 * num_half + 1, device=device
)

# Gaussian pulse
g_pulse = torch.exp(-(t_pulse**2) / (2 * sigma_t**2)) * torch.cos(
    2 * math.pi * fc * t_pulse
)

# 1-cycle square wave excitation signal at fc
num_half_exc = max(1, math.ceil(0.5 * sampling_freq / fc))
exc_signal = torch.cat(
    [torch.ones(num_half_exc, device=device), -torch.ones(num_half_exc, device=device)],
    dim=0,
)

# Convolve the excitation signal with the Gaussian pulse to get the transducer pulse-echo response h(t)
pad_len = exc_signal.numel() // 2
g_padded = torch.nn.functional.pad(g_pulse, (pad_len, pad_len))
pulse = torch.nn.functional.conv1d(
    g_padded.view(1, 1, -1), exc_signal.view(1, 1, -1).flip(-1)
).view(-1)

# Normalize the pulse energy to 1.0 to preserve gain scale
pulse = pulse / torch.linalg.norm(pulse)

# Define the forward operator with 1024 time-samples per channel, 20 MHz sampling frequency,
# signal_kind="rf" to work directly with un-demodulated real RF data, and pass the modeled pulse directly.
physics = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles,
    element_positions=ele_pos,
    n_samples=1024,
    sampling_frequency=20e6,
    sound_speed=sound_speed,
    pixel_size=pixel_size,
    signal_kind="rf",
    pulse=pulse,
    device=device,
)

# %%
# 3. Simulate Forward Channel Data (y = Ax)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We run the forward operator to simulate the raw channel-data measurements ``y``.
# For each transmit angle, a plane wave propagates down, gets scattered by our target,
# and is recorded by all 64 elements over 256 time bins.
#
# This produces a measurement tensor ``y`` of shape ``(B, 1, n_angles, n_elements, n_samples)``.
#

y = physics(x)

print("Reflectivity image shape:", x.shape)
print("Channel-data measurements shape:", y.shape)

# Let's plot the simulated raw channel-data (envelope magnitude) for the center (0 deg) transmit.
# We use our Hilbert envelope detector to remove high-frequency carrier oscillations.
envelope_y = hilbert_envelope(y[0, 0, 2].T).T
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
# 4. Reconstruct via Adjoint DAS Beamforming (x = A^T y)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The adjoint of our linearized operator implements a mathematically-rigorous,
# coherently-compounded Delay-And-Sum (DAS) beamformer directly on real RF data.
#
# Running ``physics.A_adjoint(y)`` back-projects the channel-data along the proper
# hyperbolic trajectories, compounding the angles in the real domain.
#

x_das = physics.A_adjoint(y)

# Compute the high-fidelity log-compressed B-Mode image using our Hilbert envelope detector
envelope_das = hilbert_envelope(x_das[0, 0])
db_das = 20 * torch.log10(envelope_das / envelope_das.max().clamp(min=1e-12))
bmode_das = (db_das + 40).clamp(min=0.0) / 40

# Plot the target vs the log-compressed B-Mode image. ``aspect = dz/dx`` makes
# the anisotropic pixel grid render with correct physical proportions.
dinv.utils.plot(
    [x[0], bmode_das.unsqueeze(0)],
    titles=["Ground Truth Target", "Log-Compressed B-Mode Image"],
    aspect=pixel_size[0] / pixel_size[1],
)

# %%
# 5. Explore Apodization Windows
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Apodization weights the signals from different transducer elements to control the beam shape:
#
# - **Receive Apodization**: Controlled by the F-number (`f_number`). It constrains elements within an angular acceptance cone to maintain lateral resolution and suppress clutter.
# - **Transmit Apodization**: Tapers the boundaries of the plane wave's aperture.
#
# Available window shapes: ``"rect"``, ``"hann"``, ``"hamming"``, and **``"tukey0.25"``** (Tukey window with 25% roll-off fraction).
#

# Configure an operator with receive f-number = 1.5, Hanning receive apodization,
# Tukey0.25 transmit-side aperture windowing, signal_kind="rf", and our custom pulse.
physics_apod = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles,
    element_positions=ele_pos,
    n_samples=1024,
    sampling_frequency=20e6,
    sound_speed=sound_speed,
    pixel_size=pixel_size,
    f_number=1.5,
    rx_apod_window="hann",
    tx_apod_window="tukey0.25",
    signal_kind="rf",
    pulse=pulse,
    device=device,
)

# Beamform the same channel data using our apodized operator
x_das_apod = physics_apod.A_adjoint(y)
envelope_das_apod = hilbert_envelope(x_das_apod[0, 0])
db_das_apod = 20 * torch.log10(
    envelope_das_apod / envelope_das_apod.max().clamp(min=1e-12)
)
bmode_das_apod = (db_das_apod + 40).clamp(min=0.0) / 40

dinv.utils.plot(
    [bmode_das.unsqueeze(0), bmode_das_apod.unsqueeze(0)],
    titles=["Log-Compressed B-Mode (Unapodized)", "Log-Compressed B-Mode (Apodized)"],
    aspect=pixel_size[0] / pixel_size[1],
)

# %%
# 6. Explore Interpolation Kernels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Evaluating the delay requires retrieving samples at fractional time indices.
# We support three interpolation kernels configured via the ``interp`` parameter:
#
# - ``"nearest"``: 1-tap nearest-neighbor lookup.
# - ``"linear"`` (default): 2-tap bilinear interpolation.
# - ``"keys"``: 4-tap Keys cubic convolution (equivalent to MATLAB's ``imresize('bicubic')``).
#

# Reconstruct using the high-fidelity 4-tap Keys cubic convolution kernel
physics_cubic = dinv.physics.UltrasoundPlaneWave(
    img_size=img_size,
    angles=angles,
    element_positions=ele_pos,
    n_samples=1024,
    sampling_frequency=20e6,
    sound_speed=sound_speed,
    pixel_size=pixel_size,
    interp="keys",
    signal_kind="rf",
    pulse=pulse,
    device=device,
)

x_das_cubic = physics_cubic.A_adjoint(y)
envelope_das_cubic = hilbert_envelope(x_das_cubic[0, 0])
db_das_cubic = 20 * torch.log10(
    envelope_das_cubic / envelope_das_cubic.max().clamp(min=1e-12)
)
bmode_das_cubic = (db_das_cubic + 40).clamp(min=0.0) / 40

dinv.utils.plot(
    [bmode_das.unsqueeze(0), bmode_das_cubic.unsqueeze(0)],
    titles=["Linear Interpolation B-Mode", "Keys Cubic Interpolation B-Mode"],
    aspect=pixel_size[0] / pixel_size[1],
)

# %%
# 7. Slicing Transmits / Compounding
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In self-supervised learning (such as ``Noise2Inverse``), it is often useful to separate
# the angles into subsets to train the network on independent sub-apertures.
#
# You can easily slice and select subset of transmit events using ``select_transmits()``:
#

# Keep only the -12 deg and +12 deg transmit angles (indices 0 and 4)
physics_subset = physics.select_transmits([0, 4])

y_subset = physics_subset(x)

print("Full channel data shape (5 angles):", y.shape)
print("Subset channel data shape (2 angles):", y_subset.shape)

# %%
# 8. Simulate Fully-Developed Speckle (Tissue Phantom)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In clinical ultrasound, B-mode images exhibit a characteristic grainy texture called
# **speckle**. This is caused by the constructive and destructive interference of backscattered
# waves from a high density of randomly distributed sub-wavelength structures in tissue.
#
# Here, we simulate a standard tissue phantom containing:
#
# - A dense background of randomly distributed Gaussian scatterers (fully-developed speckle).
# - A circular, non-reflective (anechoic) **cyst** in the center of the imaging field.
#

# Generate high-density random scatterers for the tissue background
torch.manual_seed(42)
background = torch.randn(1, 1, *img_size, device=device) * 0.15

# Create a circular anechoic cyst (radius = 4.3 mm) in the center of the grid.
# Because dz != dx, the physical circle is an ellipse in pixel coordinates:
# radius_z = 4.3 mm / dz ~ 112 pixels, radius_x = 4.3 mm / dx ~ 28 pixels.
zz, xx = torch.meshgrid(
    torch.arange(img_size[0], device=device),
    torch.arange(img_size[1], device=device),
    indexing="ij",
)
cyst_radius_m = 4.3e-3
rz_pix = cyst_radius_m / pixel_size[0]
rx_pix = cyst_radius_m / pixel_size[1]
dx_grid = xx - 96
dz_grid = zz - 512
cyst_mask = (dx_grid / rx_pix) ** 2 + (dz_grid / rz_pix) ** 2 <= 1.0

# Zero out reflectivity inside the cyst region
background[:, :, cyst_mask] = 0.0

# Run the forward operator on our tissue phantom
y_speckle = physics(background)

# Reconstruct via DAS compounding
x_speckle_das = physics.A_adjoint(y_speckle)
envelope_speckle = hilbert_envelope(x_speckle_das[0, 0])
db_speckle = 20 * torch.log10(
    envelope_speckle / envelope_speckle.max().clamp(min=1e-12)
)
bmode_speckle = (db_speckle + 40).clamp(min=0.0) / 40

# Plot the phantom target vs the log-compressed B-mode speckle image
dinv.utils.plot(
    [torch.abs(background[0]), bmode_speckle.unsqueeze(0)],
    titles=["Cyst Phantom Target", "Log-Compressed B-Mode Speckle Image"],
    aspect=pixel_size[0] / pixel_size[1],
)

# %%
# :References:
#
# .. footbibliography::
