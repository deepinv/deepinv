r"""
Multi-distance Fresnel phase retrieval
======================================

This example reconstructs projected phase from preprocessed
near-field holograms acquired at several propagation distances following `Huhn et al. (2022)
<https://arxiv.org/abs/2205.01099>`_: the corrected data are fitted directly
with an :math:`L_2` data term, with Tikhonov regularization or Plug-and-Play priors.
"""

# %%
# Imports and acquisition parameters
# ----------------------------------
from pathlib import Path
import numpy as np
import torch
import deepinv as dinv
import matplotlib.pyplot as plt

device = dinv.utils.get_device()

wavelength = 1.5498e-10
pixel_size = 196e-9

# %%
# Loading measured intensities
# ----------------------------


MEASUREMENTS_PATH: Path | None = (
    Path(__file__).resolve().parents[2] / "holograms_beads_updated.npz"
)

ds = 8  # downsampling factor

with np.load(MEASUREMENTS_PATH) as data:
    y = torch.from_numpy(data["holograms"]).float().to(device)
    fresnel_numbers = data["fresnelNumbers"].tolist()
    # apply downsampling
    y = y.reshape(-1, 1, y.shape[-2] // ds, ds, y.shape[-1] // ds, ds).mean(
        dim=(-3, -1)
    )
    fresnel_numbers = [fn * ds**2 for fn in fresnel_numbers]

y = y.unsqueeze(1)
height, width = y.shape[-2:]
img_size = (1, height, width)

distances = [pixel_size**2 / (wavelength * fn) for fn in fresnel_numbers]

# %%
# Projected phase model & Stacked forward model
# ---------------------------------------------

transmission = dinv.physics.Physics(A=lambda x: torch.exp(-1j * x))

# Construct stacked physics directly via list comprehension
physics = dinv.physics.stack(
    *[
        dinv.physics.compose(
            transmission,
            dinv.physics.PhaseRetrieval(
                B=dinv.physics.FresnelPropagation(
                    img_size=img_size,
                    wavelength=wavelength,
                    distance=dist,
                    pixel_size=pixel_size,
                    device=device,
                )
            ),
        )
        for dist in distances
    ]
)

dinv.utils.plot(
    list(y),
    titles=[rf"$F={fn:.2e}$" for fn in fresnel_numbers],
    figsize=(10, 3),
    cmap="gray",
    cbar=True,
    dpi=200,
    close=True,
)

# %%
# Corrected-intensity reconstruction
# ----------------------------------
data_fidelity = dinv.optim.StackedPhysicsDataFidelity(
    [dinv.optim.L2() for _ in distances]
)


class NonPositiveTikhonov(dinv.optim.Tikhonov):
    r"""Tikhonov prior whose proximal step also projects onto :math:`\phi\leq0`."""

    def prox(self, x, *args, gamma=1.0, **kwargs):
        return super().prox(x, gamma=gamma).clamp_min(0)


# Combine them in a list
reconstructor_pgd = dinv.optim.PGD(
    data_fidelity=data_fidelity,
    prior=NonPositiveTikhonov(),
    lambda_reg=3e-2,
    stepsize=0.1,
    max_iter=300,
    backtracking=dinv.optim.BacktrackingConfig(eta=0.5, max_iter=10),
)

initial_phase = torch.zeros(1, 1, height, width, device=device)
phase_estimate_pgd, metrics_pgd = reconstructor_pgd(
    y,
    physics,
    init=initial_phase,
    compute_metrics=True,
)


# %%
# Plug-and-Play (PnP) reconstruction
# ----------------------------------
# Use a pretrained denoiser as a prior inside the PGD Plug-and-Play framework.
denoiser = dinv.models.DRUNet(
    in_channels=1, out_channels=1, pretrained="download", device=device
)
prior_pnp = dinv.optim.PnP(denoiser=denoiser)


class NonPositivePnP(dinv.optim.PnP):
    r"""Tikhonov prior whose proximal step also projects onto :math:`\phi\leq0`."""

    def prox(self, x, *args, sigma_denoiser=0.1, **kwargs):
        return super().prox(x, sigma_denoiser=sigma_denoiser).clamp_min(0)


reconstructor_pnp = dinv.optim.PGD(
    data_fidelity=data_fidelity,
    prior=NonPositivePnP(denoiser=denoiser),
    stepsize=0.1,
    sigma_denoiser=0.15,
    max_iter=30,
    custom_metrics={
        "DF": lambda _values, _x_prev, x_cur: data_fidelity(x_cur, y, physics).item()
    },
)


phase_estimate_pnp, metrics_pnp = reconstructor_pnp(
    y,
    physics,
    init=initial_phase,
    compute_metrics=True,
)

# %%
# Visualization of the results
dinv.utils.plot(
    [-phase_estimate_pgd, -phase_estimate_pnp],
    titles=["Estimated phase (PGD)", "Estimated phase (PnP)"],
    figsize=(7, 3.5),
    cmap="bone",
    cbar=True,
    dpi=200,
    close=True,
)

plt.figure(figsize=(7, 3.5))
plt.plot(metrics_pgd["cost"][0], label="PGD - Loss")
plt.plot(metrics_pnp["DF"][0], label="PnP - data fidelity")
plt.legend()
plt.show()
# %%
