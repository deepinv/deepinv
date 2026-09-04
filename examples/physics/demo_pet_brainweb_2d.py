#!/usr/bin/env python3
r"""
BSREM reconstruction of a 2D BrainWeb PET slice
================================================

This example compares OSEM and BSREM with a Relative Difference Prior (RDP)
on a 2D BrainWeb PET slice containing five hot lesions. The axial slice is
extracted from the native BrainWeb volume geometry.

The reconstruction minimizes the Poisson negative log-likelihood

.. math::

    f(x) = \mathbf{1}^T(Ax+b) - y^T\log(Ax+b),

and BSREM additionally uses :class:`deepinv.optim.RDP` as :math:`\regname` in
:math:`f(x)+\lambda\reg{x}`.

.. note::

    This example requires the ``brainweb`` and ``parallelproj`` packages.
"""

# %%
import matplotlib.pyplot as plt
import parallelproj
import torch
from array_api_compat import torch as torch_compat
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.datasets import BrainWebPET
from deepinv.physics import PET

# %%
# Load a BrainWeb slice
# ---------------------
#
# ``BrainWebPET`` follows the ``(C, D, H, W)`` volume order. We select the
# middle axial slice and center-crop it in the transverse plane. A data loader
# adds the leading batch dimension expected by the physics and reconstruction
# code.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = (120, 120)


def center_slice_2d(volume):
    middle_d = volume.shape[-3] // 2
    crop_slices = tuple(
        slice((size - crop) // 2, (size + crop) // 2)
        for size, crop in zip(volume.shape[-2:], image_size, strict=True)
    )
    return volume[(..., middle_d, *crop_slices)]


dataset = BrainWebPET(subject_ids=4, transform=center_slice_2d)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
x, params = next(iter(dataloader))
x = x.to(device)

dinv.utils.plot(x, titles="BrainWeb PET activity", cbar=True)


# %%
# Add hot lesions
# ---------------
#
# All lesions have the same activity and increasing diameters, allowing us to
# study recovery coefficient as a function of lesion size. With the fixed seed,
# all five lesions intersect the selected middle slice.

lesion_diameters = [5, 8, 11, 14, 17]  # mm
lesion_dataset = BrainWebPET(
    subject_ids=4,
    transform=center_slice_2d,
    lesion_diameters=lesion_diameters,
    lesion_kwargs={
        "intensity": [192.0] * len(lesion_diameters),
        "blur": [0.0] * len(lesion_diameters),
        "thresh": 30,
    },
    seed=0,
)
lesion_dataloader = DataLoader(lesion_dataset, batch_size=1, shuffle=False)
x, params = next(iter(lesion_dataloader))
x = x.to(device)
attenuation = params["attenuation"].to(device)
lesion_mask = params["lesion_mask"].to(device)

dinv.utils.plot(
    [x, attenuation, lesion_mask],
    titles=["Emission Map", "Attenuation", "Lesions"],
    cbar=True,
)


# %%
# Simulate an attenuated PET acquisition
# --------------------------------------
#
# A single detector ring defines a 2D acquisition. As in the 3D example, we
# halve the number of endpoints per polygon side and double their spacing,
# preserving approximately the same transaxial field of view at lower sampling
# resolution. We specify the acquisition noise through a total prompt-count
# budget and a background-to-signal ratio. This makes the noise level
# independent of the normalization of the forward operator.

scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
    torch_compat,
    dev=device,
    num_rings=1,
    num_sides=34,
    num_lor_endpoints_per_side=8,
    lor_spacing=8,
)
physics = PET(
    img_size=x.shape[2:],
    voxel_size=(2, 2),
    scanner=scanner,
    fwhm_data_mm=3.0,
    gain=1.0,
    normalize=True,
    normalize_counts=True,
    device=device,
)

physics.update(attenuation=attenuation)
expected_signal = physics.A(x)

# Simulate a moderate low-count acquisition. The spatially uniform background
# is a simple approximation of random and scattered coincidences. Its total
# expected number of events is 30% of the expected true coincidences.
target_prompt_counts = 5e4
background_to_signal_ratio = 0.2
expected_background = torch.full_like(
    expected_signal,
    background_to_signal_ratio * expected_signal.mean(),
)
gain = (expected_signal.sum() + expected_background.sum()).item() / target_prompt_counts
physics.noise_model.update_parameters(gain=gain)

# The background is the expected additive rate known by the reconstruction.
# The prompt sinogram is then drawn once from the combined signal and
# background rate.
background = expected_background
physics.update(background=background)
torch.manual_seed(0)
y = physics(x)

realized_prompt_counts = round((y / gain).sum().item())
print(
    f"Expected prompt counts: {target_prompt_counts:,}; "
    f"realized: {realized_prompt_counts:,}; "
    f"background fraction: "
    f"{background_to_signal_ratio / (1 + background_to_signal_ratio):.1%}"
)

dinv.utils.plot([y], ["PET measurements"], cbar=True)


# %%
# Configure objectives and per-iteration metrics
# ------------------------------------------------

data_fidelity = dinv.optim.PoissonLikelihood(
    gain=gain,
    bkg=background / gain,
    denormalize=True,
)
rdp = dinv.optim.RDP(gamma=4.0)
lambda_reg = 0.008
# BSREM applies its update in normalized measurement units, whereas the
# objective above is evaluated in count units. Scaling the algorithmic weight
# by the gain makes both formulations have the same stationary points.
bsrem_lambda_reg = gain * lambda_reg
nrmse = dinv.metric.NRMSE()


def reconstruction_nrmse(_metrics, _x_prev, x_cur):
    return nrmse(x_cur.unsqueeze(0), x).item()


def poisson_nll(_metrics, _x_prev, x_cur):
    return data_fidelity(x_cur.unsqueeze(0), y, physics).item()


def penalized_poisson_nll(_metrics, _x_prev, x_cur):
    x_cur = x_cur.unsqueeze(0)
    return (data_fidelity(x_cur, y, physics) + lambda_reg * rdp(x_cur)).item()


metrics = {
    "nrmse": reconstruction_nrmse,
    "poisson_nll": poisson_nll,
    "penalized_poisson_nll": penalized_poisson_nll,
}


# %%
# Reconstruct with OSEM and BSREM-RDP
# -----------------------------------
#
# BSREM accepts a relaxation schedule directly. This diminishing schedule
# uses a conservative initial update and suppresses subset limit cycles more
# rapidly for this low-count acquisition.

num_subsets = 4
osem_early_iter = 3
num_iter_osem = 10
num_epochs_bsrem = 25
initialization = torch.ones_like(x)
initial_relaxation = 1
relaxation_decay = 0.8
stepsize = [
    initial_relaxation / (1.0 + relaxation_decay * k) for k in range(num_epochs_bsrem)
]
print(stepsize)

osem_early = dinv.optim.OSEM(
    data_fidelity=data_fidelity,
    num_subsets=num_subsets,
    max_iter=osem_early_iter,
)
osem = dinv.optim.OSEM(
    data_fidelity=data_fidelity,
    num_subsets=num_subsets,
    max_iter=num_iter_osem,
    custom_metrics=metrics,
    verbose=True,
    show_progress_bar=True,
)
bsrem = dinv.optim.BSREM(
    data_fidelity=data_fidelity,
    prior=rdp,
    lambda_reg=bsrem_lambda_reg,
    num_subsets=num_subsets,
    stepsize=stepsize,
    max_iter=num_epochs_bsrem,
    custom_metrics=metrics,
    verbose=True,
    show_progress_bar=True,
)

x_osem_early = osem_early(y, physics, init=initialization)
x_osem, metrics_osem = osem(y, physics, init=initialization, compute_metrics=True)
x_bsrem, metrics_bsrem = bsrem(y, physics, init=initialization, compute_metrics=True)

nrmse_osem_early = nrmse(x_osem_early, x).item()
nrmse_osem = nrmse(x_osem, x).item()
nrmse_bsrem = nrmse(x_bsrem, x).item()


# %%
# Visual comparison
# -----------------

dinv.utils.plot(
    [x, x_osem_early, x_osem, x_bsrem],
    [
        "Ground truth",
        f"OSEM ({osem_early_iter} epochs)",
        f"OSEM ({num_iter_osem} epochs)",
        f"BSREM-RDP ({num_epochs_bsrem} epochs)",
    ],
    subtitles=[
        "Reference",
        f"NRMSE: {100 * nrmse_osem_early:.2f}%",
        f"NRMSE: {100 * nrmse_osem:.2f}%",
        f"NRMSE: {100 * nrmse_bsrem:.2f}%",
    ],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    cbar=True,
    figsize=(13, 4),
)


# %%
# NRMSE along the iterates
# ------------------------

osem_epochs = range(1, len(metrics_osem["nrmse"][0]) + 1)
bsrem_epochs = range(1, len(metrics_bsrem["nrmse"][0]) + 1)
fig, axis = plt.subplots(figsize=(6, 4))
axis.plot(osem_epochs, metrics_osem["nrmse"][0], label="OSEM")
axis.plot(bsrem_epochs, metrics_bsrem["nrmse"][0], label="BSREM-RDP")
axis.axvline(
    osem_early_iter,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Early-stopped OSEM",
)
axis.set_xlabel("Epoch")
axis.set_ylabel("NRMSE")
axis.legend()
fig.tight_layout()


# %%
# Reconstruction objectives along the iterates
# ---------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(
    range(1, len(metrics_osem["poisson_nll"][0]) + 1),
    metrics_osem["poisson_nll"][0],
    label="OSEM",
)
axes[0].set_title("OSEM")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Poisson NLL")

axes[1].plot(
    range(1, len(metrics_bsrem["penalized_poisson_nll"][0]) + 1),
    metrics_bsrem["penalized_poisson_nll"][0],
    label="BSREM-RDP",
)
axes[1].set_title("BSREM-RDP")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Poisson NLL + $\\lambda$ RDP")
fig.tight_layout()


# %%
# Lesion recovery coefficients
# ----------------------------
#
# Recovery coefficient is computed independently within each labeled lesion
# mask. A value of one corresponds to perfect activity recovery.

recovery_coefficient = dinv.metric.RecoveryCoefficient()
rc_osem_early = []
rc_osem = []
rc_bsrem = []
for lesion_index in range(1, len(lesion_diameters) + 1):
    mask = lesion_mask == lesion_index
    rc_osem_early.append(recovery_coefficient(x_osem_early, x, mask=mask).item())
    rc_osem.append(recovery_coefficient(x_osem, x, mask=mask).item())
    rc_bsrem.append(recovery_coefficient(x_bsrem, x, mask=mask).item())

fig, axis = plt.subplots(figsize=(6, 4))
axis.plot(
    lesion_diameters,
    rc_osem_early,
    "o-",
    label=f"OSEM ({osem_early_iter} epochs)",
)
axis.plot(lesion_diameters, rc_osem, "o-", label=f"OSEM ({num_iter_osem} epochs)")
axis.plot(lesion_diameters, rc_bsrem, "o-", label=f"BSREM-RDP ({num_epochs_bsrem} epochs)")
axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Ideal")
axis.set_xlabel("Lesion diameter (mm)")
axis.set_ylabel("Recovery coefficient")
axis.legend()
fig.tight_layout()


# %%
# :References:
#
# .. footbibliography::

# %%
