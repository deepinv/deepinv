#!/usr/bin/env python
r"""
3D PET listmode simulation and reconstruction
==============================================

This example simulates a 3D NEMA IEC phantom acquisition, displays listmode
and sinogram representations of the events, and compares listmode and sinogram
reconstruction, with and without time-of-flight (ToF) information.

.. note::

    This example requires the ``parallelproj`` package. Install it with
    ``pixi install -e full`` or ``conda install -c conda-forge parallelproj``;
    see :ref:`install`. A CUDA GPU is recommended for the 3D acquisition.

The events are simulated by expanding integer sinogram counts. Their order does
not represent acquisition time, and they do not contain timestamps. Real
listmode acquisitions require scanner-specific decoding and calibration.
"""

# %%
# sphinx_gallery_tags = ["Tomography", "PET/SPECT", "Low-photon imaging"]

import time

import matplotlib.pyplot as plt
import torch
import parallelproj
from array_api_compat import torch as torch_compat

import deepinv as dinv
from deepinv.physics import PET
from deepinv.utils import NEMAIECPhantomDataset

# %%
# Phantom and scanner
# -------------------
# The NEMA IEC dataset provides a 3D emission phantom and an attenuation map.
# We scale emission to control the number of events; attenuation remains in
# physical units of mm\ :sup:`-1`. The phantom grid spans 320 mm transversely
# and 220 mm axially. The 65 × 96 × 96 grid gives nearly isotropic voxels,
# improving the representation of the cylindrical insert and small spheres.
#
# This 24-ring scanner is a simplified geometry rather than a model of a
# particular clinical system. Each ring has 16 sides with 14 detectors per side.

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
img_size = (65, 96, 96)  # (D, H, W)
voxel_size = (220 / (img_size[0] - 1), 320 / (img_size[1] - 1), 320 / (img_size[2] - 1))
sample = NEMAIECPhantomDataset(size=img_size, use_dict_output=True, return_labels=True)[
    0
]
x = 0.30 * sample["x"].unsqueeze(0).to(device)
attenuation = sample["params"]["attenuation"].unsqueeze(0).to(device)
labels = sample["params"]["labels"].unsqueeze(0).to(device)
sphere_slice = round((27 + 110) / voxel_size[0])

dinv.utils.plot(
    [x[:, :, sphere_slice], attenuation[:, :, sphere_slice]],
    titles=["Emission image", r"Attenuation map (mm$^{-1}$)"],
    figsize=(8, 4),
    cbar=True,
)

scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    torch_compat,
    dev=device,
    radius=380.0,
    num_sides=16,
    num_lor_endpoints_per_side=14,
    lor_spacing=10.7,
    ring_positions=torch.linspace(-110, 110, 24, device=device),
    symmetry_axis=2,
)
physics = PET(
    img_size,
    voxel_size=voxel_size,
    scanner=scanner,
    device=device,
    attenuation=attenuation,
)
physics.plot_geometry()

# %%
# Simulating a non-ToF acquisition
# --------------------------------
# We first use the sinogram model to simulate integer counts with unit gain.
# A small, uniform expected background represents events not originating from
# the emission image. Attenuation and background are included both when
# simulating measurements and when reconstructing.
#
# Each sinogram bin represents one line of response (LOR) between a pair of
# detectors. Expanding its integer count yields one event per count. The returned
# indices record the original sinogram bin for each event; this lets us bin the
# events back without losing counts.

with torch.no_grad():
    physics.update(background=torch.full_like(physics.A(x), 0.02))
    counts = physics(x)
_, _, _, indices = dinv.physics.sinogram_to_listmode(physics, counts[0, 0])
sinogram = dinv.physics.listmode_to_sinogram(physics, indices)
assert torch.equal(sinogram, counts[0, 0].long())
print(f"Sinogram shape: {tuple(counts.shape)}")
print(f"Simulated {indices.numel():,} events; binning reproduces the sinogram.")

# Each event stores detector endpoint coordinates in mm. The list is sorted by
# bin here because it was expanded from a sinogram; an acquisition's event order
# and timestamps cannot be recovered from those counts.
example_bin = counts[0, 0].reshape(-1).argmax()
example_events = (indices == example_bin).nonzero().flatten()
print(
    f"Sinogram bin {example_bin.item()} contains {example_events.numel()} counts: "
    f"event rows {example_events.tolist()}."
)

# Build the event projector with matching attenuation and background.
lm, event_weights = dinv.physics.listmode_physics_from_sinogram(physics, counts[0, 0])
dinv.utils.print_pet_listmode(lm, num_events=5)

# Compare the same ring-pair plane in the simulated sinogram and the sinogram
# constructed from its listmode events. Both use the same Poisson realization.
sinogram_plane = counts.shape[-1] // 2
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for axis, image, title in zip(
    axes,
    (counts[0, 0, ..., sinogram_plane], sinogram[..., sinogram_plane]),
    ("Simulated sinogram", "Sinogram from listmode"),
    strict=True,
):
    sino_image = axis.imshow(
        image.cpu(), cmap="gray", aspect="auto", vmin=0, vmax=counts.max().item()
    )
    axis.set(xlabel="View", ylabel="Radial bin", title=title)
fig.colorbar(sino_image, ax=axes, label="Counts")

# %%
# Listmode and sinogram MLEM
# --------------------------
# A listmode projector evaluates the forward model only at observed events.
# Attenuation and background are gathered at the same LORs. The full sinogram
# physics provides the sensitivity :math:`A^\top \mathbf{1}`, including LORs
# where no event was detected. The event projector's adjoint of ones is *not*
# the full-acquisition sensitivity.
#
# Sinogram and listmode MLEM below use the same events, initialization, background,
# attenuation and iteration count. They should give the same result up to
# numerical precision. We time the reconstruction calls, including sensitivity
# calculation and metric evaluation, but excluding simulation and conversion.
# Their relative speed depends on both the number of sinogram bins and the
# number of events: listmode processes repeated events separately, while a
# sinogram stores repeated events in the same bin.

initial = torch.full_like(x, x.mean().item())
nrmse = dinv.metric.NRMSE()
custom_metrics = {
    "nrmse": lambda _values, _x_prev, x_cur: nrmse(x_cur.unsqueeze(0), x).item()
}


def sync():
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


# Warm both projection paths before timing, including GPU kernel initialization.
with torch.no_grad():
    physics.A_adjoint(physics.A(initial))
    lm.A_adjoint(lm.A(initial))
sync()

mlem_iter = 12
sinogram_model = dinv.optim.MLEM(
    data_fidelity=dinv.optim.PoissonLikelihood(bkg=physics.background),
    max_iter=mlem_iter,
    custom_metrics=custom_metrics,
)
listmode_model = dinv.optim.MLEM(
    data_fidelity=dinv.optim.PoissonLikelihood(bkg=lm.background),
    max_iter=mlem_iter,
    custom_metrics=custom_metrics,
)
with torch.no_grad():
    sync()
    begin = time.perf_counter()
    x_sinogram, metrics_sinogram = sinogram_model(
        counts, physics, init=initial, compute_metrics=True
    )
    sync()
    sinogram_time = time.perf_counter() - begin
    begin = time.perf_counter()
    x_listmode, metrics_listmode = listmode_model(
        event_weights, lm, init=initial, compute_metrics=True
    )
    sync()
    listmode_time = time.perf_counter() - begin

relative_difference = (x_sinogram - x_listmode).norm() / x_sinogram.norm()
print(
    f"Relative difference between the two MLEM reconstructions: {relative_difference.item():.2e}"
)

# %%
# Sinogram and listmode OSEM
# --------------------------
# OSEM applies updates over subsets of the acquisition. The sinogram model
# splits views into angular subsets, while the listmode model randomly assigns
# event numbers to subsets and uses an equal fraction of the full sensitivity
# for each subset. Both use the same events, attenuation, background, initial
# image, subset count and number of epochs. Because their subset assignments
# differ, their updates and results are not identical, and the runtime comparison
# reflects these different partitioning strategies. Each epoch performs one
# update per subset, so an epoch is not the same as an MLEM iteration.

num_subsets, subset_epochs = 4, 3
sinogram_osem = dinv.optim.OSEM(
    num_subsets=num_subsets, max_iter=subset_epochs, custom_metrics=custom_metrics
)
listmode_osem = dinv.optim.OSEM(
    num_subsets=num_subsets,
    max_iter=subset_epochs,
    custom_metrics=custom_metrics,
)
with torch.no_grad():
    sync()
    begin = time.perf_counter()
    x_osem_sinogram, metrics_osem_sinogram = sinogram_osem(
        counts, physics, init=initial, compute_metrics=True
    )
    sync()
    osem_sinogram_time = time.perf_counter() - begin
    begin = time.perf_counter()
    x_osem, metrics_osem = listmode_osem(
        event_weights, lm, init=initial, compute_metrics=True
    )
    sync()
    osem_time = time.perf_counter() - begin

# %%
# Reconstruction comparison
# -------------------------
reconstructions = [x_sinogram, x_listmode, x_osem_sinogram, x_osem]
names = [
    "Sinogram MLEM",
    "Listmode MLEM",
    "Sinogram OSEM",
    "Listmode OSEM",
]
metrics_all = [
    metrics_sinogram,
    metrics_listmode,
    metrics_osem_sinogram,
    metrics_osem,
]
dinv.utils.plot(
    [x[:, :, sphere_slice], *(image[:, :, sphere_slice] for image in reconstructions)],
    titles=["Reference", *names],
    subtitles=[
        "",
        *(
            f"NRMSE: {100 * nrmse(image, x).item():.1f}" + r"$\%$"
            for image in reconstructions
        ),
    ],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    figsize=(16, 4),
)

# Compare convergence by iteration or epoch. The sinogram and listmode likelihoods
# differ by a data-dependent constant, so plot changes from the first reported
# cost rather than their absolute values.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, metrics, marker in zip(names, metrics_all, ("o", "s", "^", "D"), strict=True):
    cost = metrics["cost"][0]
    style = "--" if name.startswith("Sinogram") else "-"
    axes[0].plot(
        range(1, len(cost) + 1),
        [value - cost[0] for value in cost],
        style,
        marker=marker,
        label=name,
    )
    errors = metrics["nrmse"][0]
    axes[1].plot(
        range(1, len(errors) + 1),
        [100 * value for value in errors],
        style,
        marker=marker,
        label=name,
    )
axes[0].set(xlabel="Iteration / epoch", ylabel="Change in Poisson objective")
axes[1].set(xlabel="Iteration / epoch", ylabel="Volume NRMSE (%)")
for ax in axes:
    ax.legend(fontsize=8)
fig.tight_layout()

# %%
# Sphere recovery coefficients
# ----------------------------
# Recovery is the reconstructed activity inside each sphere divided by its
# ground-truth activity. A coefficient of one indicates perfect recovery.

sphere_diameters = [10, 13, 17, 22, 28, 37]
recovery_coefficient = dinv.metric.RecoveryCoefficient()
fig, axis = plt.subplots(figsize=(8, 5))
for name, reconstruction in zip(names, reconstructions, strict=True):
    recovery = [
        recovery_coefficient(reconstruction, x, mask=labels == i).item()
        for i in range(1, len(sphere_diameters) + 1)
    ]
    axis.plot(sphere_diameters, recovery, marker="o", label=name)
axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Ideal")
axis.set_xlabel("Sphere diameter (mm)")
axis.set_ylabel("Recovery coefficient")
axis.legend()
fig.tight_layout()

# Release the non-ToF measurements and event projector before allocating ToF data.
del counts, sinogram, image, indices, example_events, event_weights, lm
del sinogram_model, listmode_model, sinogram_osem, listmode_osem

# %%
# Time-of-flight listmode measurements
# ------------------------------------
# ToF adds a signed bin to each event, giving an estimate of its position along
# the LOR. We simulate a separate ToF acquisition with the same phantom,
# scanner and expected total background. Summing the ToF bins produces the
# corresponding non-ToF counts. We compare sinogram and listmode
# reconstructions of the ToF acquisition. This is a new Poisson realization,
# distinct from the non-ToF acquisition above.
#
# A finite ToF range and truncated kernel can make the summed ToF operator
# slightly different from the non-ToF operator; we report that difference. ToF
# may converge faster but also become noisier with further iterations. Seven
# ToF bins reduce the memory needed for the 3D sinogram.

tof_info = parallelproj.tof.TOFParameters(
    num_tofbins=9, tofbin_width=80.0, sigma_tof=20.0, num_sigmas=3.0
)
physics_tof = PET(
    img_size,
    voxel_size=voxel_size,
    scanner=scanner,
    device=device,
    tof_info=tof_info,
    attenuation=physics.attenuation.unsqueeze(-1),
    background=physics.background.unsqueeze(-1) / tof_info.num_tofbins,
)
with torch.no_grad():
    counts_tof = physics_tof(x)
    expected_tof = physics_tof.A(x).sum(dim=-1)
    expected_non_tof = physics.A(x)
    mismatch = (expected_tof - expected_non_tof).abs().sum() / expected_non_tof.sum()
lm_tof, weights_tof = dinv.physics.listmode_physics_from_sinogram(
    physics_tof, counts_tof[0, 0]
)
print(
    f"ToF sinogram bins: {counts_tof.numel():,}; listmode events: {weights_tof.shape[-1]:,}; "
    f"summed-ToF/non-ToF projection difference: {100 * mismatch.item():.2f}%"
)
print(f"First five signed ToF bins: {lm_tof.event_proj.event_tofbins[:5].tolist()}")
del expected_tof, expected_non_tof

# %%
# ToF sinogram and listmode reconstruction
# ----------------------------------------
# As in the non-ToF comparison, the timings include sensitivity computation
# and metrics but exclude simulation and conversion. MLEM uses the same
# acquisition in both representations. OSEM uses angular subsets for the
# sinogram and random event subsets for listmode. ToF uses fewer MLEM
# iterations than non-ToF, with matching counts for each representation.
#
# ToF listmode projection can be faster because it evaluates only the observed
# ToF bin of each event, whereas a sinogram projector evaluates all ToF bins.
# The complete reconstruction timings also include work outside projection,
# including sensitivity and subset construction.

# Warm all projection paths before timing, including the sinogram ToF path.
with torch.no_grad():
    physics_tof.A_adjoint(physics_tof.A(initial))
    lm_tof.A_adjoint(lm_tof.A(initial))
sync()

tof_mlem_iter = 6
tof_subset_epochs = 2
tof_sinogram_mlem = dinv.optim.MLEM(
    data_fidelity=dinv.optim.PoissonLikelihood(bkg=physics_tof.background),
    max_iter=tof_mlem_iter,
    custom_metrics=custom_metrics,
)
tof_listmode_mlem = dinv.optim.MLEM(
    data_fidelity=dinv.optim.PoissonLikelihood(bkg=lm_tof.background),
    max_iter=tof_mlem_iter,
    custom_metrics=custom_metrics,
)
with torch.no_grad():
    sync()
    begin = time.perf_counter()
    x_tof_sinogram_mlem, metrics_tof_sinogram_mlem = tof_sinogram_mlem(
        counts_tof, physics_tof, init=initial, compute_metrics=True
    )
    sync()
    tof_sinogram_mlem_time = time.perf_counter() - begin
    begin = time.perf_counter()
    x_tof_listmode_mlem, metrics_tof_listmode_mlem = tof_listmode_mlem(
        weights_tof, lm_tof, init=initial, compute_metrics=True
    )
    sync()
    tof_listmode_mlem_time = time.perf_counter() - begin

tof_mlem_difference = (x_tof_sinogram_mlem - x_tof_listmode_mlem).norm() / (
    x_tof_sinogram_mlem.norm()
)
print(f"ToF MLEM relative difference: {tof_mlem_difference.item():.2e}")

tof_sinogram_osem = dinv.optim.OSEM(
    num_subsets=num_subsets, max_iter=tof_subset_epochs, custom_metrics=custom_metrics
)
tof_osem = dinv.optim.OSEM(
    num_subsets=num_subsets,
    max_iter=tof_subset_epochs,
    custom_metrics=custom_metrics,
)
with torch.no_grad():
    sync()
    begin = time.perf_counter()
    x_tof_sinogram, metrics_tof_sinogram = tof_sinogram_osem(
        counts_tof, physics_tof, init=initial, compute_metrics=True
    )
    sync()
    tof_sinogram_osem_time = time.perf_counter() - begin
    begin = time.perf_counter()
    x_tof, metrics_tof = tof_osem(
        weights_tof, lm_tof, init=initial, compute_metrics=True
    )
    sync()
    tof_time = time.perf_counter() - begin

tof_reconstructions = (
    ("ToF sinogram MLEM", x_tof_sinogram_mlem),
    ("ToF listmode MLEM", x_tof_listmode_mlem),
    ("ToF sinogram OSEM", x_tof_sinogram),
    ("ToF listmode OSEM", x_tof),
)

dinv.utils.plot(
    [
        x[:, :, sphere_slice],
        *(image[:, :, sphere_slice] for _, image in tof_reconstructions),
    ],
    titles=["Reference", *(name for name, _ in tof_reconstructions)],
    subtitles=[
        "",
        *(
            f"NRMSE: {100 * nrmse(image, x).item():.1f}" + r"$\%$"
            for _, image in tof_reconstructions
        ),
    ],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    figsize=(16, 4),
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, metrics, marker in (
    ("ToF sinogram MLEM", metrics_tof_sinogram_mlem, "o"),
    ("ToF listmode MLEM", metrics_tof_listmode_mlem, "s"),
):
    errors = metrics["nrmse"][0]
    axes[0].plot(
        range(1, len(errors) + 1),
        [100 * value for value in errors],
        marker=marker,
        label=name,
    )
for name, metrics, marker in (
    ("ToF sinogram OSEM", metrics_tof_sinogram, "^"),
    ("ToF listmode OSEM", metrics_tof, "D"),
):
    errors = metrics["nrmse"][0]
    axes[1].plot(
        range(1, len(errors) + 1),
        [100 * value for value in errors],
        marker=marker,
        label=name,
    )
axes[0].set(xlabel="MLEM iteration", ylabel="Volume NRMSE (%)")
axes[1].set(xlabel="OSEM epoch", ylabel="Volume NRMSE (%)")
for ax in axes:
    ax.legend()
fig.tight_layout()

# %%
# Reconstruction timings
# ----------------------
# Each time includes sensitivity computation and metric evaluation.
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
timing_labels = ["MLEM\nsinogram", "MLEM\nlistmode", "OSEM\nsinogram", "OSEM\nlistmode"]
timings = (
    (sinogram_time, listmode_time, osem_sinogram_time, osem_time),
    (tof_sinogram_mlem_time, tof_listmode_mlem_time, tof_sinogram_osem_time, tof_time),
)
for axis, title, times in zip(
    axes,
    (
        f"Non-ToF: {mlem_iter} MLEM iterations, {subset_epochs} OSEM epochs",
        f"ToF: {tof_mlem_iter} MLEM iterations, {tof_subset_epochs} OSEM epochs",
    ),
    timings,
    strict=True,
):
    bars = axis.bar(timing_labels, times)
    axis.bar_label(bars, fmt="%.3g", padding=3)
    axis.set_title(title)
axes[0].set_ylim(0, max(max(times) for times in timings) * 1.2)
axes[0].set_ylabel("Reconstruction time (s)")
fig.tight_layout()

# %%
# ToF sphere recovery coefficients
# --------------------------------
# Compare sphere recovery for the same ToF acquisition.
fig, axis = plt.subplots(figsize=(8, 5))
for name, reconstruction in tof_reconstructions:
    recovery = [
        recovery_coefficient(reconstruction, x, mask=labels == i).item()
        for i in range(1, len(sphere_diameters) + 1)
    ]
    axis.plot(sphere_diameters, recovery, marker="o", label=name)
axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Ideal")
axis.set_xlabel("Sphere diameter (mm)")
axis.set_ylabel("Recovery coefficient")
axis.legend()
fig.tight_layout()
plt.show()

# %%
