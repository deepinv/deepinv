#!/usr/bin/env python3
r"""
OSEM, BSREM and mirror descent for 3D BrainWeb PET
==================================================

This example compares OSEM, BSREM and mirror descent with a Relative Difference
Prior (RDP) on a BrainWeb PET phantom containing five hot lesions. The native
BrainWeb volume geometry matches the Siemens Biograph mMR reconstruction grid.

The reconstruction minimizes the Poisson negative log-likelihood

.. math::

    f(x) = \mathbf{1}^T(Ax+b) - y^T\log(Ax+b),

and both BSREM and mirror descent additionally use :class:`deepinv.optim.RDP`
as :math:`\regname` in :math:`f(x)+\lambda\reg{x}`.

.. note::

    This is a large 3D example and is intended to run on a CUDA-capable
    machine. It requires the ``brainweb`` and ``parallelproj`` packages.
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
# Load a BrainWeb volume
# ----------------------
#
# ``BrainWebPET`` follows the ``(C, D, H, W)`` volume order. A data loader adds
# the leading batch dimension expected by the physics and reconstruction code.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
volume_size = (120, 120, 120)


def center_crop_3d(volume):
    crop_slices = tuple(
        slice((size - crop) // 2, (size + crop) // 2)
        for size, crop in zip(volume.shape[-3:], volume_size, strict=True)
    )
    return volume[(..., *crop_slices)]


dataset = BrainWebPET(subject_ids=4, transform=center_crop_3d)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
x, params = next(iter(dataloader))
x = x.to(device)

dinv.utils.plot_ortho3D(x, titles="Emission Map", figsize=(4, 4))


# %%
# Add hot lesions
# ---------------
#
# All lesions have the same activity and increasing diameters, allowing us to
# study recovery coefficient as a function of lesion size.

lesion_diameters = [5, 8, 11, 14, 17]  # mm
lesion_dataset = BrainWebPET(
    subject_ids=4,
    transform=center_crop_3d,
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

dinv.utils.plot_ortho3D(
    [x, attenuation, lesion_mask],
    titles=["Emission Map", "Attenuation", "Lesions"],
    figsize=(12, 4),
)


# %%
# Simulate an attenuated PET acquisition
# --------------------------------------
#
# The reduced scanner supplied by :class:`deepinv.physics.PET` uses 16 rings,
# whose axial field of view is too narrow for this brain volume. Here we use
# parallelproj's full 36-ring demo geometry, whose approximately 195 mm axial
# extent covers the nonzero part of the 120-voxel crop much more closely. To
# limit GPU memory, we halve the number of endpoints per polygon side and
# double their spacing, preserving approximately the same transaxial field of
# view at lower sampling resolution. We specify the acquisition noise through
# a total prompt-count budget and a background-to-signal ratio. This makes the
# noise level independent of the normalization of the forward operator.

scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
    torch_compat,
    dev=device,
    num_sides=34,
    num_lor_endpoints_per_side=8,
    lor_spacing=8,
)
physics = PET(
    img_size=x.shape[2:],
    voxel_size=(2, 2, 2),
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
target_prompt_counts = 5e6
background_to_signal_ratio = 0.3
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

# Plot one sinogram plane after adding attenuation and background.
dinv.utils.plot(
    [y[..., y.shape[-1] // 2]],
    ["PET measurements"],
    cbar=True,
    figsize=(3, 4),
)


# %%
# Configure objectives and per-iteration metrics
# ------------------------------------------------

data_fidelity = dinv.optim.PoissonLikelihood(
    gain=gain,
    bkg=background / gain,
    denormalize=True,
)
rdp = dinv.optim.RDP(gamma=2.0)
lambda_reg = 0.002
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

num_subsets = 8
osem_early_iter = 3
osem_iter = 10
bsrem_iter = 30
initialization = torch.ones_like(x)
initial_relaxation = 1
relaxation_decay = 0.9
stepsize = [
    initial_relaxation / (1.0 + relaxation_decay * k) for k in range(bsrem_iter)
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
    max_iter=osem_iter,
    custom_metrics=metrics,
    verbose=True,
    show_progress_bar=True,
)
bsrem = dinv.optim.BSREM(
    data_fidelity=data_fidelity,
    prior=rdp,
    lambda_reg=lambda_reg,
    num_subsets=num_subsets,
    stepsize=stepsize,
    max_iter=bsrem_iter,
    custom_metrics=metrics,
    verbose=True,
    show_progress_bar=True,
)

x_osem_early = osem_early(y, physics, init=initialization)
x_osem, metrics_osem = osem(y, physics, init=initialization, compute_metrics=True)
x_bsrem, metrics_bsrem = bsrem(y, physics, init=initialization, compute_metrics=True)

# %%
# Reconstruct with entropy mirror descent
# ----------------------------------------
#
# DeepInv's general-purpose MD solver can minimize the same RDP-regularized
# Poisson objective. Negative entropy gives multiplicative updates, preserving
# positivity from our strictly positive initialization without a projection.
# Unlike OSEM and BSREM, each MD iteration uses the full sinogram once. We use
# fewer iterations than in the 2D example because each 3D pass is much costlier.
# Both the operator and measurements are expressed in photon-count units so
# that the likelihood uses unit gain; the background enters only the fidelity.

count_physics = dinv.physics.LinearPhysics(
    A=lambda z: physics.A(z) / gain,
    A_adjoint=lambda z: physics.A_adjoint(z) / gain,
)
count_fidelity = dinv.optim.PoissonLikelihood(gain=1.0, bkg=background / gain)
md_iter = 30
md = dinv.optim.MD(
    bregman_potential=dinv.optim.NegEntropy(),
    data_fidelity=count_fidelity,
    prior=rdp,
    lambda_reg=lambda_reg,
    stepsize=1.0,
    max_iter=md_iter,
    custom_metrics=metrics,
    verbose=True,
    show_progress_bar=True,
)
x_md, metrics_md = md(
    y / gain, count_physics, init=initialization, compute_metrics=True
)

nrmse_osem_early = nrmse(x_osem_early, x).item()
nrmse_osem = nrmse(x_osem, x).item()
nrmse_bsrem = nrmse(x_bsrem, x).item()
nrmse_md = nrmse(x_md, x).item()


# %%
# Visual comparison
# -----------------
#
# We display the middle axial slice of each volume.

middle_d = x.shape[2] // 2
dinv.utils.plot(
    [
        x[:, :, middle_d],
        x_osem_early[:, :, middle_d],
        x_osem[:, :, middle_d],
        x_bsrem[:, :, middle_d],
        x_md[:, :, middle_d],
    ],
    [
        "Ground truth",
        f"OSEM ({osem_early_iter} epochs)",
        f"OSEM ({osem_iter} epochs)",
        f"BSREM-RDP ({bsrem_iter} epochs)",
        f"MD-RDP ({md_iter} iterations)",
    ],
    subtitles=[
        "Reference",
        f"NRMSE: {100 * nrmse_osem_early:.2f}%",
        f"NRMSE: {100 * nrmse_osem:.2f}%",
        f"NRMSE: {100 * nrmse_bsrem:.2f}%",
        f"NRMSE: {100 * nrmse_md:.2f}%",
    ],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    cbar=True,
    figsize=(16, 4),
)


# %%
# NRMSE along the iterates
# ------------------------
#
# Omit the initial MD transient so that it does not compress the vertical scale
# of the later iterates.

md_plot_start = 4
osem_epochs = range(1, len(metrics_osem["nrmse"][0]) + 1)
bsrem_epochs = range(1, len(metrics_bsrem["nrmse"][0]) + 1)
fig, axis = plt.subplots(figsize=(8, 5))
axis.plot(osem_epochs, metrics_osem["nrmse"][0], label="OSEM")
axis.plot(bsrem_epochs, metrics_bsrem["nrmse"][0], label="BSREM-RDP")
axis.plot(
    range(md_plot_start, len(metrics_md["nrmse"][0]) + 1),
    metrics_md["nrmse"][0][md_plot_start - 1 :],
    label=f"MD-RDP (from iteration {md_plot_start})",
)
axis.axvline(
    osem_early_iter,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Early-stopped OSEM",
)
axis.set_xlabel("Full-data passes (epoch / MD iteration)")
axis.set_ylabel("NRMSE")
axis.legend()
fig.tight_layout()


# %%
# Reconstruction objectives along the iterates
# ---------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
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
axes[1].plot(
    range(md_plot_start, len(metrics_md["penalized_poisson_nll"][0]) + 1),
    metrics_md["penalized_poisson_nll"][0][md_plot_start - 1 :],
    label=f"MD-RDP (from iteration {md_plot_start})",
)
axes[1].set_title("RDP-regularized reconstruction")
axes[1].legend()
axes[1].set_xlabel("Full-data passes (epoch / MD iteration)")
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
rc_md = []
for lesion_index in range(1, len(lesion_diameters) + 1):
    mask = lesion_mask == lesion_index
    rc_osem_early.append(recovery_coefficient(x_osem_early, x, mask=mask).item())
    rc_osem.append(recovery_coefficient(x_osem, x, mask=mask).item())
    rc_bsrem.append(recovery_coefficient(x_bsrem, x, mask=mask).item())
    rc_md.append(recovery_coefficient(x_md, x, mask=mask).item())

fig, axis = plt.subplots(figsize=(8, 5))
axis.plot(
    lesion_diameters,
    rc_osem_early,
    "o-",
    label=f"OSEM ({osem_early_iter} epochs)",
)
axis.plot(lesion_diameters, rc_osem, "o-", label=f"OSEM ({osem_iter} epochs)")
axis.plot(lesion_diameters, rc_bsrem, "o-", label=f"BSREM-RDP ({bsrem_iter} epochs)")
axis.plot(lesion_diameters, rc_md, "o-", label=f"MD-RDP ({md_iter} iterations)")
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
