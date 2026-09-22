r"""
Multi-distance Fresnel phase retrieval
======================================

This example reconstructs projected phase from preprocessed
near-field holograms acquired at several propagation distances, see e.g. :footcite:t:`huhn2022Fast`
for details on the physics.
"""

# %%
import torch
import deepinv as dinv
import matplotlib.pyplot as plt

device = dinv.utils.get_device()

# %%
# Get physical parameters and load data
# -------------------------------------------

wavelength = 1.5498e-10
pixel_size = 196e-9

# downsampling factor for computational feasibility. Set to 1 for full resolution
ds = 8

# load data
data = dinv.utils.load_example("holograms_multidistance_hotopy.npz", grayscale=True)
y = data["holograms"].to(device)
fresnel_numbers = data["fresnelNumbers"].tolist()

# apply downsampling
y = y.reshape(-1, 1, y.shape[-2] // ds, ds, y.shape[-1] // ds, ds).mean(dim=(-3, -1))
fresnel_numbers = [fn * ds**2 for fn in fresnel_numbers]

height, width = y.shape[-2:]
img_size = (1, height, width)

distances = [pixel_size**2 / (wavelength * fn) for fn in fresnel_numbers]

# %%
# The Fresnel number :math:`a_F` is defined as
#
# .. math::
#
#   a_F = \frac{a^2}{\lambda d},
#
# where :math:`a` is
# the characteristic size of the object,
# typically the pixel size,
# :math:`\lambda` the wavelength, and :math:`d` the propagation distance.
# Hence, it is also affected by downsampling the data.

# visualize the measurements
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
# Define transmission model & stacked forward model
# --------------------------------------------------
# Via the transmission model, we decide how the object is mapped
# to the complex field.
#
# Usually the object consists of a
# phase-shiftig part :math:`\phi \leq 0` and an absorption part :math:`\mu \geq 0` .
# In this example, we follow :footcite:t:`huhn2022Fast` and assume
# that the object is purely phase-shifting, i.e. :math:`\mu=0`, and thus the transmission
# model which is generally given by :math:`A(\phi) = \exp(i \phi - \mu)` simplifies to
# :math:`A(\phi) = \exp(i \phi)`.
#
# For computational reasons we reconstruct :math:`-\phi`
# instead of :math:`\phi`, which slightly changes the code.

transmission = dinv.physics.Physics(A=lambda x: torch.exp(-1j * x))

# Collect shared arguments of the Fresnel propagation
prop_kwargs = {
    "img_size": img_size,
    "wavelength": wavelength,
    "pixel_size": pixel_size,
    "device": device,
}

# Build the stacked physics step-by-step for each distance
models = []
for dist in distances:
    fresnel = dinv.physics.FresnelPropagation(distance=dist, **prop_kwargs)
    phase_retrieval = dinv.physics.PhaseRetrieval(B=fresnel)
    single_physics = dinv.physics.compose(transmission, phase_retrieval)
    models.append(single_physics)

physics = dinv.physics.stack(*models)

# %%
# Reconstruction using proximal gradient descent (PGD) and Tikhonov prior
# -------------------------------------------------------------------------
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
# Using DRUNet as a pretrained denoiser inside the PGD Plug-and-Play framework.
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
# Visualization of the results and convergence
# ----------------------------------------------
# Flipping the sign again. Note that with the default downsampling the results are not comparable
# to :footcite:t:`huhn2022Fast` which were computed at full resolution on a powerful GPU.
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
# :References:
#
# .. footbibliography::
