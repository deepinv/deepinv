r"""
CT tomography with a pretrained Diffusers model
=====================================================================

This demo shows an end-to-end DeepInv workflow with a pretrained diffusion model hosted on
HuggingFace:

- Load a *volume* (here: a small synthetic CT volume provided by DeepInv) as a stack of 2D slices.
- Simulate (parallel-beam) X-ray projections using :class:`deepinv.physics.Tomography`.
- Draw an unconditional sample from the diffusion prior.
- Perform posterior sampling / reconstruction from projections using diffusion posterior sampling (DPS).

.. note::
    This example uses the pixel-space diffusion UNet from:
    https://huggingface.co/jiayangshi/lodochallenge_pixel_diffusion

    The model is a 2D UNet trained on 1-channel 512x512 CT slices with intensities rescaled to ``[-1, 1]``.

.. warning::
    Running this demo can be compute- and download-heavy on CPU.

    Install requirements:

    - ``pip install diffusers transformers``

"""

# %%
import torch
import torch.nn.functional as F
import deepinv as dinv

from deepinv.models import DiffusersDenoiserWrapper
from deepinv.physics import Tomography
from deepinv.sampling import (
    PosteriorDiffusion,
    EulerSolver,
    VariancePreservingDiffusion,
    DPSDataFidelity,
)
from deepinv.optim import ZeroFidelity

# %% Configuration
# This demo can be memory hungry (512x512 diffusion model). For a robust out-of-the-box
# experience, default to CPU. Switch to CUDA manually if you have enough free VRAM.
device = "cuda"
dtype = torch.float32

# The DM4CT pixel diffusion model is trained for 512x512, 1-channel.
img_width = 512
# Number of slices in the volume (increase for a larger volume).
n_slices = 1

# Tomography configuration (keep small for a quick demo)
angles = 30
noise_std = 0.01

# %%
# Load an example CT slice and simulate X-ray projections with DeepInv physics
x = dinv.utils.load_example("CT100_256x256_0.pt", img_size=img_width).to(device)
x = F.interpolate(
    x, size=(img_width, img_width), mode="bilinear", align_corners=False, antialias=True
)

physics = dinv.physics.TomographyWithAstra(
    img_size=(img_width, img_width),
    angles=angles,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_std),
)

y = physics(x)

dinv.utils.plot(
    [x, y],
    titles=["Ground truth slice", "Simulated projections"],
    figsize=(8, 4),
)

# %% 3) Load the pretrained diffusion model from HuggingFace via Diffusers
# -----------------------------------------------------------------------
# The wrapper exposes the diffusers UNet as a DeepInv denoiser.
denoiser = DiffusersDenoiserWrapper(
    model_id="jiayangshi/lodochallenge_pixel_diffusion",
    device=device,
)


# %%
# Generates an unconditional slice sample from the diffusion prior.
num_steps = 100
timesteps = torch.linspace(1.0, 1e-3, num_steps, device=device)
solver = EulerSolver(timesteps=timesteps, rng=torch.Generator(device=device))

sde = VariancePreservingDiffusion(
    denoiser=denoiser,
    solver=solver,
    dtype=torch.float64,
    device=device,
)

# prior_sampler = PosteriorDiffusion(
#     data_fidelity=ZeroFidelity(),
#     sde=sde,
#     denoiser=denoiser,
#     solver=solver,
#     dtype=torch.float32,
#     device=device,
#     verbose=True,
# )

# with torch.no_grad():
#     x_prior_slices = []
#     for i in range(n_slices):
#         x_prior_slices.append(
#             prior_sampler(
#                 y=None,
#                 physics=None,
#                 x_init=(1, 1, img_width, img_width),
#                 seed=i,
#                 get_trajectory=False,
#             )
#         )
#     x_prior = torch.cat(x_prior_slices, dim=0)

# dinv.utils.plot(
#     [x_prior],
#     titles=["Unconditional prior sample"],
#     figsize=(4, 4),
# )

# %% 5) Reconstruction / posterior sampling from projections (DPS)
# ---------------------------------------------------------------
# We use diffusion posterior sampling to condition the diffusion prior on the measurements.
num_steps = 100
timesteps = torch.linspace(1.0, 1e-3, num_steps, device=device)
solver = EulerSolver(timesteps=timesteps, rng=torch.Generator(device=device))

sde = VariancePreservingDiffusion(
    denoiser=denoiser,
    solver=solver,
    dtype=torch.float64,
    device=device,
)

posterior_sampler = PosteriorDiffusion(
    data_fidelity=DPSDataFidelity(denoiser=denoiser, weight=1.0),
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    dtype=torch.float32,
    device=device,
    verbose=True,
)

with torch.no_grad():
    x_rec_slices = []
    for i in range(n_slices):
        x_rec_slices.append(
            posterior_sampler(
                y=y[i : i + 1],
                physics=physics,
                x_init=(1, 1, img_width, img_width),
                seed=100 + i,
                get_trajectory=False,
            )
        )
    x_rec = torch.cat(x_rec_slices, dim=0)

# %%
dinv.utils.plot(
    [x_rec],
    titles=["DPS posterior sample"],
    figsize=(4, 4),
)
# %%
# Show the results

x_fbp = physics.A_dagger(y)

dinv.utils.plot(
    [
        x,
        x_fbp,
        x_rec,
    ],
    titles=["Ground truth", "FBP", "DPS posterior sample"],
    figsize=(10, 3),
)

# %%
