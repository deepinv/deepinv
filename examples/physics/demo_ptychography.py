r"""
Ptychography phase retrieval
============================

Ptychography is a coherent imaging technique for recovering a complex-valued
object, including both its amplitude and phase, from intensity-only
measurements. A localized illumination, called the *probe*, is scanned across
overlapping regions of the object. At each position, a detector records the
intensity after the exit wave has propagated from the object plane. The overlap
provides redundant measurements of the same object regions, making it possible
to recover phase information that is not measured directly by the detector.

In far-field ptychography, propagation to the detector is described by a
Fourier transform :math:`\mathcal{F}`. For a known probe :math:`p`, the noiseless
measurement at scan position :math:`s_\ell` is

.. math::

    y_\ell = \left|\mathcal{F}\left(p \odot x_\ell\right)\right|^2,

where :math:`x_\ell` is the probe-sized patch of the complex object at
:math:`s_\ell`, and :math:`p \odot x_\ell` is the exit wave leaving the sample.

In this example, we use two images to define the amplitude and phase of a
complex object. We then build a complex probe, set up the far-field geometry,
and simulate diffraction patterns with :class:`deepinv.physics.Ptychography`.
Finally, we reconstruct the object from these measurements.
"""

# %%
# General setup
# -------------
# We import the libraries used below and select a GPU if one is available,
# or the CPU otherwise.

import matplotlib.pyplot as plt
import torch

import deepinv as dinv
from deepinv.optim.data_fidelity import AmplitudeLoss
from deepinv.optim.phase_retrieval import correct_global_phase
from deepinv.physics import FarFieldPtychographyGeometry, Ptychography
from deepinv.utils import load_example

device = dinv.utils.get_device()

# %%
# Load toy images to create a target object
# -----------------------------
# We take one color channel from each of two images, using one for the object's
# amplitude and the other for its phase.

size = 128
amplitude_image = load_example("butterfly.png", grayscale=False, img_size=(size, size))
phase_image = load_example("CBSD_0010.png", grayscale=False, img_size=(size, size))

x_amplitude = amplitude_image[:, 0, ...].unsqueeze(1)  # Take only one channel
x_phase = phase_image[:, 0, ...].unsqueeze(1)
print(x_amplitude.shape, x_phase.shape)
dinv.utils.plot(
    [x_amplitude, x_phase],
    titles=["Amplitude image", "Phase image"],
    figsize=(6, 3),
)
# %%
# Prepare the complex object
# --------------------------
# We combine the images into a complex transmission function, with amplitude
# values in [0.3, 1] and phase values in :math:`[-\pi/2, \pi/2]`.

# Keep the amplitude above zero so the phase remains observable.
amplitude_min = 0.3
amplitude = amplitude_min + (1 - amplitude_min) * x_amplitude / x_amplitude.max()
phase = torch.pi * (x_phase / x_phase.max() - 0.5)  # between -pi/2 and pi/2
x = (amplitude * torch.exp(1j * phase.to(torch.complex64))).to(device)

# %%
# Set up the physical geometry
# ----------------------------
# We define the far-field geometry using the illumination wavelength, the
# sample-to-detector distance, and the detector pixel size after binning.
# Together with the detector shape, these determine the object-plane pixel
# size through the Fraunhofer relation. We use this pixel size to convert
# physical distances to pixels.

img_size = (1, size, size)
probe_size = 64  # detector and diffraction-pattern shape, in pixels
probe_shape = (1, probe_size, probe_size)
native_detector_pixel_size = 4.5e-6
detector_binning = 8
effective_detector_pixel_size = detector_binning * native_detector_pixel_size

geometry = FarFieldPtychographyGeometry(
    wavelength=632.8e-9,  # visible light
    sample_detector_distance=5e-2,
    detector_shape=probe_shape[-2:],
    detector_pixel_size=(
        effective_detector_pixel_size,
        effective_detector_pixel_size,
    ),
)
object_dy, object_dx = geometry.object_pixel_size
object_fov = torch.tensor(geometry.object_extent(img_size[-2:]))  # (height, width)
print(f"Object-plane pixel size: ({object_dy * 1e6:.2f}, {object_dx * 1e6:.2f}) um")
print(
    f"Object field of view: ({object_fov[0] * 1e6:.1f}, {object_fov[1] * 1e6:.1f}) um"
)

# %%
# Build the probe
# ---------------
# We specify the probe radius in metres and convert it to pixels using the
# object-plane pixel size. We then create a circular probe with
# :func:`deepinv.physics.phase_retrieval.build_probe` and add a quadratic phase
# profile to model a curved wavefront, as produced by a thin lens. The phase
# increases from zero at the centre to :math:`\pi` at the edge of the aperture.

probe_radius_m = 4e-4  # illuminated radius on the sample
probe_radius = round(probe_radius_m / object_dx)  # in pixels (isotropic geometry)
print(f"Probe radius: {probe_radius_m * 1e6:.1f} um = {probe_radius} pixels")

probe = dinv.physics.phase_retrieval.build_probe(
    probe_shape, type="disk", probe_radius=probe_radius, device=device
)
# Centre the phase profile on the disk so its phase ranges from zero to pi.
coordinates = torch.arange(probe_size, device=device) - probe_size // 2
yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
lens_phase = torch.pi * (xx**2 + yy**2) / probe_radius**2
probe = probe.to(torch.complex64) * torch.exp(1j * lens_phase)

# We plot the magnitude in grayscale and the phase with the cyclic twilight
# colormap, clipped to [-pi, pi] so that its colorbar reads in radians.
fig, axs = plt.subplots(1, 2, figsize=(7, 3), squeeze=False, layout="tight")
dinv.utils.plot(
    probe.abs(),
    titles="Probe magnitude",
    rescale_mode=None,
    cbar=True,
    fig=fig,
    axs=axs[:, :1],
    show=False,
)
dinv.utils.plot(
    probe.angle(),
    titles="Probe phase (rad)",
    cmap="twilight",
    rescale_mode="clip",
    vmin=-torch.pi,
    vmax=torch.pi,
    cbar=True,
    fig=fig,
    axs=axs[:, 1:],
)

# %%
# Define the scanning grid in physical units
# ----------------------------------
# We choose the scan spacing from the desired overlap between neighbouring
# probe positions. For a probe of diameter :math:`d` and an overlap fraction
# :math:`o`, the spacing is :math:`(1 - o) d`. The overlap provides the redundant
# measurements needed for phase retrieval.
# We define the grid in metres and extend it far enough for the probe to reach
# the object corners.
#
# For experimental data, replace this grid with the stage positions from the
# scan file, stored as an ``(N, 2)`` array in metres, in ``(row, column)`` order.

target_overlap = 0.7
scan_step = (1 - target_overlap) * 2 * probe_radius_m
scan_span = object_fov - torch.sqrt(torch.tensor(2.0)) * probe_radius_m

side_n_img = int(torch.ceil(scan_span / scan_step).max()) + 1
scan_rows = torch.linspace(-scan_span[0] / 2, scan_span[0] / 2, side_n_img)
scan_cols = torch.linspace(-scan_span[1] / 2, scan_span[1] / 2, side_n_img)
positions = torch.cartesian_prod(scan_rows, scan_cols)
print(
    f"Scan: {len(positions)} positions, {scan_step * 1e6:.1f} um apart, "
    f"spanning {scan_span[0] * 1e6:.1f} um"
)

# %%
# Convert the scan to pixel shifts and build the operator
# -------------------------------------------------------
# :meth:`deepinv.physics.PtychographyGeometry.positions_to_shifts`
# divides the stage positions by the object-plane pixel size and rounds to the
# nearest pixel to obtain the shifts used by the operator.

shifts = geometry.positions_to_shifts(positions)
n_img = shifts.shape[0]
pixel_step = torch.diff(torch.unique(shifts[:, 0])).max()
print(f"Scan step: {scan_step * 1e6:.1f} um = {pixel_step} pixels")

physics = Ptychography(
    img_size=img_size,
    probe=probe,
    shifts=shifts,
    device=device,
    geometry=geometry,
)

# %%
# Display probe overlap
# ---------------------
# We display the overlap for two consecutive probe positions and for the full
# scan to see how the probes cover the object.

overlap_img = physics.B.get_overlap_img(physics.B.shifts).cpu()
probe_index = n_img // 2
overlap2probe = physics.B.get_overlap_img(
    physics.B.shifts[probe_index : probe_index + 2]
).cpu()
dinv.utils.plot(
    [overlap2probe.unsqueeze(0), overlap_img.unsqueeze(0)],
    titles=["Overlap 2 probe", "Overlap images"],
)


# %%
# Simulate the measurements
# -------------------------
# Applying the operator gives one diffraction pattern per scan position.
# We show the first four patterns, which come from the first row of the scan
# grid. Neighbouring probes illuminate overlapping regions, so the speckle
# pattern changes gradually between positions.

y = physics(x)
print(f"Measurements: {tuple(y.shape)} (batch, positions, detector rows, columns)")

# ``fftshift`` to move the zero frequency from the corner to the centre of each image and
# log scale for clearly showing the range of intensities
patterns = torch.fft.fftshift(y[0, :4], dim=(-2, -1)).log()
dinv.utils.plot(
    list(patterns.unsqueeze(1)),
    titles=[f"Position {i + 1} (log)" for i in range(len(patterns))],
    figsize=(10, 3),
)


# %%
# Optimize the amplitude loss
# ---------------------------
# We start with an object of uniform amplitude and zero phase, then use Adam
# to minimize the amplitude loss. Both the object's amplitude and phase are
# free to vary during reconstruction. Note that one could also extend the current demo to
# the blind ptychography case of reconstructing the probe simultaneously.

data_fidelity = AmplitudeLoss()
n_iter = 350
x_est = torch.ones_like(x, requires_grad=True)
optimizer = torch.optim.Adam([x_est], lr=0.05)
loss_hist = []

for i in range(n_iter):
    optimizer.zero_grad()
    loss = data_fidelity(x_est, y, physics).mean()
    loss.backward()
    optimizer.step()
    loss_hist.append(loss.detach().cpu())
    if i % 10 == 0:
        print(f"Iter {i}, loss: {loss.item():.2e}")

# Plot the loss curve
plt.plot(loss_hist)
plt.yscale("log")
plt.title("Amplitude loss")
plt.show()

# %%
# Compare the reconstruction with the original object
# --------------------------------------------------
# Correct the global phase offset and compare the ground-truth and
# estimated amplitude and phase.


x_est = x_est.detach().cpu()
final_est = correct_global_phase(x_est, x.cpu())

# Use the same range and normalization for the original and reconstructed
# images so their colours can be compared directly.
fig, axs = plt.subplots(1, 2, figsize=(7, 3), squeeze=False, layout="tight")
dinv.utils.plot(
    {"Ground-truth amplitude": amplitude, "Estimated amplitude": final_est.abs()},
    rescale_mode=None,
    vmin=0,
    vmax=1,
    cbar=True,
    fig=fig,
    axs=axs,
)

fig, axs = plt.subplots(1, 2, figsize=(7, 3), squeeze=False, layout="tight")
dinv.utils.plot(
    {
        "Ground-truth phase (rad)": phase,
        "Estimated phase (rad)": torch.angle(final_est),
    },
    rescale_mode="clip",
    vmin=-torch.pi / 2,
    vmax=torch.pi / 2,
    cbar=True,
    fig=fig,
    axs=axs,
)
