r"""
Tour of ultrafast ultrasound in DeepInverse
===========================================

This example presents the plane-wave ultrafast ultrasound forward physics available in
DeepInverse for pulse-echo imaging problems:

-  Physics: :class:`deepinv.physics.UltrafastUltrasound`,
   :class:`deepinv.physics.UltrasoundPlaneWave`

Contents:

1. The acquisition setup
2. The pulse-echo impulse response
3. Defining the forward operator
4. Simulating per-channel raw data
5. Beamforming with the adjoint
6. Coherent plane-wave compounding
7. Receive apodization

"""

import math

import numpy as np
import torch
from scipy.signal import hilbert

import deepinv as dinv

device = dinv.utils.get_device()

# %%
# 1. The acquisition setup
# ------------------------
#
# As a first step, let's simulate the following pulse-echo experiment:
# A 64-element linear array at 0.3 mm pitch fires 11 plane waves spread over
# :math:`\pm 12^\circ`. The region to image is given in meters -- 5 to 40 mm deep, 24 mm
# wide.

n_elements, pitch = 64, 3e-4
element_x = (torch.arange(n_elements) - (n_elements - 1) / 2) * pitch
element_positions = torch.stack(
    [element_x, torch.zeros(n_elements)], dim=-1).to(device)
aperture = float(element_x.max() - element_x.min())

angles = torch.linspace(math.radians(-12.0),
                        math.radians(12.0), 11, device=device)

sound_speed, center_frequency, sampling_frequency = 1540.0, 5e6, 20e6
wavelength = sound_speed / center_frequency

depth_min, depth_max, width = 5e-3, 40e-3, 24e-3
pixel_size = (wavelength / 6, wavelength / 2)
pixel_origin = (depth_min, -width / 2)
img_size = (
    round((depth_max - depth_min) / pixel_size[0]),
    round(width / pixel_size[1]),
)

# %%
# 2. The pulse-echo impulse response
# ----------------------------------
#
# In order to account for the physics of a transducer element, we simulate a typical
# pulse-echo impulse response as a Gaussian-modulated pulse with a given fractional
# bandwidth and center frequency. This pulse-echo impulse response models the response of a
# transducer element i.e. how acoustical signals are transformed to electrical signals and
# vice-versa.

fractional_bandwidth = 0.8
sigma_t = math.sqrt(2 * math.log(2)) / (
    math.pi * fractional_bandwidth * center_frequency
)
n_half = math.ceil(3.5 * sigma_t * sampling_frequency)
t_pulse = torch.arange(-n_half, n_half + 1, device=device) / sampling_frequency
pulse = torch.exp(-(t_pulse**2) / (2 * sigma_t**2)) * torch.cos(
    2 * math.pi * center_frequency * t_pulse
)
pulse = pulse / torch.linalg.norm(pulse)

# %%
# 3. Defining the forward operator
# --------------------------------
#
# :class:`deepinv.physics.UltrasoundPlaneWave` gathers everything above: the grid to
# reconstruct on (``img_size``, ``pixel_size``, ``pixel_origin``), the sequence
# (``angles``), the probe (``element_positions``, ``sampling_frequency``, ``sound_speed``,
# ``pulse``) and the beamforming settings (``f_number``, ``receive_apod_window``, the
# subject of the last section).

longest_path = math.hypot(depth_max, width / 2) + math.hypot(
    depth_max, (width + aperture) / 2
)
n_samples = math.ceil(longest_path / sound_speed *
                      sampling_frequency) + pulse.numel()

operator_args = dict(
    img_size=img_size,
    angles=angles,
    element_positions=element_positions,
    n_samples=n_samples,
    sampling_frequency=sampling_frequency,
    sound_speed=sound_speed,
    pixel_size=pixel_size,
    pixel_origin=pixel_origin,
    signal_kind="rf",
    pulse=pulse,
    normalize=False,
    device=device,
)
physics = dinv.physics.UltrasoundPlaneWave(
    **operator_args, f_number=1.5, receive_apod_window="hann"
)

print(
    f"imaged region: {depth_min * 1e3:.0f}-{depth_max * 1e3:.0f} mm deep, ", end="")
print(
    f"{width * 1e3:.0f} mm wide, sampled on a {img_size[0]}x{img_size[1]} grid")

# %%
# 4. Simulating per-channel raw data
# ----------------------------------
#
# We simulate the per-channel raw data using the operator defined before. To do so, we
# consider a reflectivity map x composed of 3 points located at (0, 15mm), (-7.5mm, 25mm)
# and (7.5mm, 35mm).

x = torch.zeros(1, 1, *img_size, device=device)
grid = physics.pixel_grid
spots = []
for depth_mm, lateral_mm in ((15.0, 0.0), (25.0, -7.5), (35.0, 7.5)):
    position = torch.tensor([lateral_mm, depth_mm], device=grid.device) * 1e-3
    distance = torch.linalg.vector_norm(grid - position, dim=-1)
    row, column = divmod(int(distance.argmin()), img_size[1])
    x[0, 0, row, column] = 1.0
    spots.append((row, column))

y = physics(x)
print("reflectivity image:", tuple(x.shape))
print("channel data:      ", tuple(y.shape))


DYNAMIC_RANGE = 40.0
envelope = torch.from_numpy(np.abs(hilbert(y.cpu().numpy(), axis=0)))
db = 20 * torch.log10(envelope / envelope.max().clamp(min=1e-12))
bmode_channel = (db + DYNAMIC_RANGE).clamp(min=0.0) / DYNAMIC_RANGE


dinv.utils.plot(
    bmode_channel[:, :, 0],
    titles=[r"Channel data, transmit at $-12^\circ$"],
    figsize=(20, 4),
)

# %%
# 5. Beamforming with the adjoint
# -------------------------------
#
# We build an estimate of the reflectivity map (the traditional DAS image) by applying the
# adjoint operator to the per-channel raw data.

x_das = physics.A_adjoint(y)

envelope = torch.from_numpy(np.abs(hilbert(x_das.cpu().numpy(), axis=-2)))
db = 20 * torch.log10(envelope / envelope.max().clamp(min=1e-12))
bmode_das = (db + DYNAMIC_RANGE).clamp(min=0.0) / DYNAMIC_RANGE

dinv.utils.plot(
    [x, bmode_das],
    titles=["Scatterers", f"Beamformed, {angles.numel()} transmits"],
    aspect=pixel_size[0] / pixel_size[1],
    figsize=(10, 10),
)

# %%
# 6. Coherent plane-wave compounding
# ----------------------------------
#
# We demonstrate the effect of coherent plane wave compounding, a well known technique used
# in ultrafast ultrasound imaging to improve image quality. Here, we show an image
# reconstructed with 1 PW and an image reconstructed with 11 PWs. This allows us to show
# how the function "select_transmits" works in practice.

center = len(angles) // 2
physics_1pw = physics.select_transmits([center])
x_1pw = physics_1pw.A_adjoint(y[:, :, center: center + 1])

envelope = torch.from_numpy(np.abs(hilbert(x_1pw.cpu().numpy(), axis=-2)))
db = 20 * torch.log10(envelope / envelope.max().clamp(min=1e-12))
bmode_1pw = (db + DYNAMIC_RANGE).clamp(min=0.0) / DYNAMIC_RANGE

dinv.utils.plot(
    [bmode_1pw, bmode_das],
    titles=["1 transmit", f"{angles.numel()} transmits"],
    aspect=pixel_size[0] / pixel_size[1],
    figsize=(10, 10),
)
