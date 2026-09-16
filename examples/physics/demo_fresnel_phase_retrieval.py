r"""
Multi-distance Fresnel phase retrieval
======================================

This example reconstructs projected phase from preprocessed
near-field holograms acquired at several propagation distances following`Huhn et al. (2022)
<https://arxiv.org/abs/2205.01099>`_: the corrected data are fitted directly
with an :math:`L_2` data term, with Tikhonov regularization or plug and play priors.
"""

# %%
# Imports and acquisition parameters
# ----------------------------------
#
# The parameters match the four-distance polystyrene-microsphere data described in
# table 1 of the paper: 8 keV X-rays and a 196 nm effective pixel size. After
# the cone-beam holograms are rescaled to a common magnification, their Fresnel
# numbers define the equivalent plane-wave propagation distances used here via
# :math:`z=\Delta x^2/(\lambda F)`.
from pathlib import Path

import numpy as np
import torch

import deepinv as dinv

device = dinv.utils.get_device()

RESULTS_DIR = Path("results") / "fresnel_phase_retrieval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

wavelength = 1.5498e-10
pixel_size = 196e-9


# %%
# Using measured intensities
# --------------------------
#
# ``holograms_beads_updated.npz`` contains the already dark- and flat-field-
# corrected holograms under ``holograms`` and their matching Fresnel numbers
# under ``fresnelNumbers``. The different distances have already been registered
# and rescaled to a common magnification, and the incident intensity is
# normalized to one. No calibration fields or raw detector counts are needed.
MEASUREMENTS_PATH: Path | None = (
    Path(__file__).resolve().parents[2] / "holograms_beads_updated.npz"
)


with np.load(MEASUREMENTS_PATH) as data:
    corrected_intensity = torch.from_numpy(data["holograms"]).float().to(device)
    fresnel_numbers = tuple(data["fresnelNumbers"].tolist())

measurements = dinv.utils.TensorList(
    [measurement[None, None] for measurement in corrected_intensity]
)
height, width = corrected_intensity.shape[-2:]


distances = tuple(
    pixel_size**2 / (wavelength * fresnel_number) for fresnel_number in fresnel_numbers
)
img_size = (1, height, width)


# %%
# Projected phase model
# ---------------------
#
# We use the convention of the paper, where the phase is given by
#
# .. math::
#
#     \phi=-k\bar\delta\leq0,
#
# where :math:`k=2\pi/\lambda`. In the pure-phase approximation the plane-wave
# transmission is :math:`T=\exp(\mathrm{i}\phi)`. Absorption is therefore set
# to zero rather than reconstructed as an independent image.


def phase_to_transmission(phase, **kwargs):
    return torch.exp(1j * phase)


# %%
# Stacked forward model
# ---------------------
#
# For ideal plane-wave illumination, :math:`p=1`. For each distance, DeepInv
# composes the nonlinear transmission, linear Fresnel propagation, and
# intensity detection:
#
# .. math::
#
#     \mathcal A_j(\phi)
#     =\left|P_{z_j}[e^{i\phi}]\right|^2.
#
# The paper supplies corrected relative intensities rather than raw counts, so
# the forward operators are deterministic and contain no detector noise model.
transmission = dinv.physics.Physics(A=phase_to_transmission)
plane_physics = []

for distance in distances:
    propagation = dinv.physics.FresnelPropagation(
        img_size=img_size,
        wavelength=wavelength,
        distance=distance,
        pixel_size=pixel_size,
        device=device,
    )
    intensity = dinv.physics.PhaseRetrieval(B=propagation)
    plane_physics.append(dinv.physics.compose(transmission, intensity))

physics = dinv.physics.stack(*plane_physics)


dinv.utils.plot(
    list(measurements),
    titles=[rf"$F={fresnel_number:.2e}$" for fresnel_number in fresnel_numbers],
    save_fn=RESULTS_DIR / "measurements.png",
    figsize=(10, 3),
    cmap="gray",
    cbar=True,
    dpi=200,
    close=True,
)


# %%
# Corrected-intensity reconstruction
# ----------------------------------
#
# Following equation (7) of the paper, the corrected holograms are fitted with
# squared :math:`\ell_2` residuals under the negative-phase constraint:
#
# .. math::
#
#     \widehat\phi
#     =\underset{\phi\leq0}{\operatorname{argmin}}\;
#     \frac12\sum_j\|\mathcal A_j(\phi)-y_j\|_2^2
#     +\frac\alpha2\|\phi\|_2^2.
#
# The paper uses different Tikhonov weights for different frequency bands;
# this compact example uses DeepInv's scalar Tikhonov prior.
data_fidelity = dinv.optim.StackedPhysicsDataFidelity(
    [dinv.optim.L2() for _ in distances]
)


class NonPositiveTikhonov(dinv.optim.Tikhonov):
    r"""Tikhonov prior whose proximal step also projects onto :math:`\phi\leq0`."""

    def prox(self, x, *args, gamma=1.0, **kwargs):
        return super().prox(x, gamma=gamma).clamp_max(0)


# The projection implements the paper's negative-phase range constraint. No
# support constraint is imposed. Tikhonov regularization fixes the otherwise
# unobservable spatially constant phase, so no mean subtraction is required.
prior = NonPositiveTikhonov()
initial_phase = torch.zeros(1, 1, height, width, device=device)

reconstructor = dinv.optim.PGD(
    data_fidelity=data_fidelity,
    prior=prior,
    lambda_reg=1e-3,
    stepsize=0.4,
    max_iter=300,
    backtracking=dinv.optim.BacktrackingConfig(eta=0.5, max_iter=10),
)

phase_estimate, metrics = reconstructor(
    measurements,
    physics,
    init=initial_phase,
    compute_metrics=True,
)


# %%
# Results
# -------
images = [phase_estimate]
titles = [r"Estimated phase ($\phi$)"]


dinv.utils.plot(
    images,
    titles=titles,
    save_fn=RESULTS_DIR / "reconstruction.png",
    figsize=(4 * len(images), 3.5),
    cmap="bone",
    cbar=True,
    dpi=200,
    close=True,
)

cost = np.asarray(metrics["cost"][0])
dinv.utils.plot_curves(
    {"residual": [(cost).tolist()]},
    save_dir=RESULTS_DIR / "objective_history",
)
print(f"Saved figures to {RESULTS_DIR.resolve()}")

# %%
