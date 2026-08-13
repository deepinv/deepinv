#!/usr/bin/env python3
r"""
BSREM reconstruction of a 3D BrainWeb PET volume
=================================================

This example compares OSEM and BSREM with a Relative Difference Prior (RDP)
on a BrainWeb PET phantom containing five hot lesions. The native BrainWeb
volume geometry matches the Siemens Biograph mMR reconstruction grid.

The reconstruction minimizes the Poisson negative log-likelihood

.. math::

    f(x) = \mathbf{1}^T(Ax+b) - y^T\log(Ax+b),

and BSREM additionally uses :class:`deepinv.optim.RDP` as :math:`\regname` in
:math:`f(x)+\lambda\reg{x}`.

.. note::

    This is a large 3D example and is intended to run on a CUDA-capable
    machine. It requires the ``brainweb`` and ``parallelproj`` packages.
"""

# %%
import matplotlib.pyplot as plt
import parallelproj
import torch
from array_api_compat import torch as torch_compat

import deepinv as dinv
from deepinv.datasets import BrainWebPET
from deepinv.physics import PET


# %%
# Load a BrainWeb volume with five lesions
# -----------------------------------------
#
# All lesions have the same activity and increasing diameters, allowing us to
# study recovery coefficient as a function of lesion size. ``BrainWebPET``
# returns PET-ready arrays in ``(C, H, W, D)`` order, so only a batch dimension
# is added.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lesion_diameters = [5, 8, 11, 14, 17]  # mm
volume_size = (120, 120, 120)


def center_crop_3d(volume):
    crop_slices = tuple(
        slice((size - crop) // 2, (size + crop) // 2)
        for size, crop in zip(volume.shape[-3:], volume_size, strict=True)
    )
    return volume[(..., *crop_slices)]


dataset = BrainWebPET(
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
emission, params = dataset[0]
x = emission.unsqueeze(0).to(device)
attenuation = params["attenuation"].unsqueeze(0).to(device)
lesion_mask = params["lesion_mask"].unsqueeze(0).to(device)

# Display the three orthogonal middle slices of the ground-truth activity.
middle_h, middle_w, middle_d = (size // 2 for size in x.shape[-3:])
dinv.utils.plot(
    [
        x[..., middle_d],
        x[:, :, middle_h, :, :].transpose(-2, -1),
        x[:, :, :, middle_w, :].transpose(-2, -1),
    ],
    ["Axial", "Coronal", "Sagittal"],
    figsize=(9, 4),
    cbar=True,
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
# view at lower sampling resolution. The background is generated as an
# independent Poisson realization, as in the general 3D PET demo.

gain = 1 / 1000
scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
    torch_compat,
    dev=device,
    num_lor_endpoints_per_side=8,
    lor_spacing=8.0625,
)
physics = PET(
    img_size=x.shape[2:],
    voxel_size=(2.0863, 2.0863, 2.03125),
    scanner=scanner,
    fwhm_data_mm=4.3,
    gain=gain,
    normalize=False,
    normalize_counts=False,
    device=device,
)

physics.update(attenuation=attenuation)
expected_signal = physics.A(x)
expected_background = torch.ones_like(expected_signal) * x.max() * 0.05
background = physics.generate_background(expected_background)
physics.update(background=background)
y = physics(x)

# Plot one sinogram plane after adding attenuation and background.
dinv.utils.plot(
    [y[..., y.shape[-1] // 2]],
    ["PET measurements"],
    cbar=True,
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
lambda_reg = 0.03
nrmse = dinv.metric.NRMSE()


# %%
# Reconstruct with OSEM and BSREM-RDP
# -----------------------------------
#
# BSREM accepts a relaxation schedule directly. This diminishing schedule
# preserves large early updates and gradually suppresses subset limit cycles.

num_subsets = 8
num_iter = 10
initialization = torch.ones_like(x)
stepsize = [1.0 / (1.0 + 0.2 * k) for k in range(num_iter)]

osem = dinv.optim.OSEM(
    data_fidelity=data_fidelity,
    num_subsets=num_subsets,
    max_iter=num_iter,
    custom_metrics={
        "nrmse": lambda _values, _x_prev, x_cur: nrmse(x_cur.unsqueeze(0), x).item(),
        "penalized_poisson": lambda _values, _x_prev, x_cur: (
            data_fidelity(x_cur.unsqueeze(0), y, physics)
            + lambda_reg * rdp(x_cur.unsqueeze(0))
        ).item(),
    },
)
bsrem = dinv.optim.BSREM(
    data_fidelity=data_fidelity,
    prior=rdp,
    lambda_reg=lambda_reg,
    num_subsets=num_subsets,
    stepsize=stepsize,
    max_iter=num_iter,
    custom_metrics={
        "nrmse": lambda _values, _x_prev, x_cur: nrmse(x_cur.unsqueeze(0), x).item(),
        "poisson": lambda _values, _x_prev, x_cur: data_fidelity(
            x_cur.unsqueeze(0), y, physics
        ).item(),
    },
)

x_osem, metrics_osem = osem(y, physics, init=initialization, compute_metrics=True)
x_bsrem, metrics_bsrem = bsrem(y, physics, init=initialization, compute_metrics=True)

nrmse_osem = nrmse(x_osem, x).item()
nrmse_bsrem = nrmse(x_bsrem, x).item()


# %%
# Visual comparison
# -----------------
#
# We display the middle axial slice of each volume.

dinv.utils.plot(
    [
        x[..., middle_d],
        x_osem[..., middle_d],
        x_bsrem[..., middle_d],
    ],
    ["Ground truth", "OSEM", "BSREM-RDP"],
    subtitles=[
        "Reference",
        f"NRMSE: {100 * nrmse_osem:.2f}%",
        f"NRMSE: {100 * nrmse_bsrem:.2f}%",
    ],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    cbar=True,
    figsize=(10, 4),
)


# %%
# Evolution of reconstruction quality and objectives
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
iterations = range(1, num_iter + 1)

axes[0].plot(iterations, metrics_osem["nrmse"][0], label="OSEM")
axes[0].plot(iterations, metrics_bsrem["nrmse"][0], label="BSREM-RDP")
axes[0].set_ylabel("NRMSE")

axes[1].plot(iterations, metrics_osem["cost"][0], label="OSEM")
axes[1].plot(iterations, metrics_bsrem["poisson"][0], label="BSREM-RDP")
axes[1].set_ylabel("Poisson NLL")

axes[2].plot(
    iterations,
    metrics_osem["penalized_poisson"][0],
    label="OSEM",
)
axes[2].plot(iterations, metrics_bsrem["cost"][0], label="BSREM-RDP")
axes[2].set_ylabel("Poisson NLL + RDP")

for axis in axes:
    axis.set_xlabel("Epoch")
    axis.legend()
fig.tight_layout()


# %%
# Lesion recovery coefficients
# ----------------------------
#
# Recovery coefficient is computed independently within each labeled lesion
# mask. A value of one corresponds to perfect activity recovery.

recovery_coefficient = dinv.metric.RecoveryCoefficient()
rc_osem = []
rc_bsrem = []
for lesion_index in range(1, len(lesion_diameters) + 1):
    mask = lesion_mask == lesion_index
    rc_osem.append(recovery_coefficient(x_osem, x, mask=mask).item())
    rc_bsrem.append(recovery_coefficient(x_bsrem, x, mask=mask).item())

fig, axis = plt.subplots(figsize=(6, 4))
axis.plot(lesion_diameters, rc_osem, "o-", label="OSEM")
axis.plot(lesion_diameters, rc_bsrem, "o-", label="BSREM-RDP")
axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Ideal")
axis.set_xlabel("Lesion diameter (mm)")
axis.set_ylabel("Recovery coefficient")
axis.legend()
fig.tight_layout()


# %%
# :References:
#
# .. footbibliography::
