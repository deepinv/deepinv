r"""
Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse
======================================================================================

This demo shows you how to use our wrapper
:class:`deepinv.models.DiffusersDenoiserWrapper` to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.

See more about the `diffusers pipeline <https://huggingface.co/docs/diffusers/index>`_ and our posterior sampling `user guide <https://deepinv.github.io/deepinv/auto_examples/sampling/demo_diffusion_sde.html>`_.

.. note:: This example requires the `diffusers` and `transformers` package. You can install it via `pip install diffusers transformers`.

"""

# %%
import torch
import deepinv as dinv
from deepinv.models.wrapper import DiffusersDenoiserWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
figsize = 2.5

from deepinv.sampling import (
    PosteriorDiffusion,
    EulerSolver,
    VarianceExplodingDiffusion,
    VariancePreservingDiffusion,
)
from deepinv.optim import ZeroFidelity

# %% Load a pretrained model and wrap it as a denoiser
# ----------------------------------------------------
#
# Let us first load a pretrained diffusion model from the HuggingFace Hub. Here, we use the `google/ddpm-ema-celebahq-256` model.
# This model is trained on 256x256 of CelebA dataset using the DDPM scheduler.

# We can wrap any diffusers model as a DeepInv denoiser using one line of code:
denoiser = DiffusersDenoiserWrapper(
    mode_id="google/ddpm-ema-celebahq-256", device=device
)

# Load an example image
x = dinv.utils.load_example(
    "celeba_example2.jpg",
    img_size=256,
    resize_mode="resize",
).to(device)

# Add noise and test the denoiser
sigma = 0.1
x_noisy = x + sigma * torch.randn_like(x)
with torch.no_grad():
    x_denoised = denoiser(x_noisy, sigma=sigma)

dinv.utils.plot(
    [x, x_noisy, x_denoised],
    figsize=(figsize * 3, figsize),
    titles=["Original image", "Noisy image", "Denoised image"],
)

# %% Unconditional image generation
# ---------------------------------
#
# It is also possible to use the wrapped model for unconditional image generation.
# The model was trained with DDPM scheduler, however we can use it with any SDE provided in DeepInv.
# Here, we use the Variance Exploding SDE with Euler solver for sampling.

num_steps = 125
rng = torch.Generator(device)
timesteps = torch.linspace(1, 0.001, num_steps)
solver = EulerSolver(timesteps=timesteps, rng=rng)

sigma_min = 0.001
sigma_max = 80
sde = VarianceExplodingDiffusion(
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    alpha=0.5,
    device=device,
    dtype=dtype,
)

model = PosteriorDiffusion(
    data_fidelity=ZeroFidelity(),
    sde=sde,
    denoiser=denoiser,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
)
sample, trajectory = model(
    y=None,
    physics=None,
    x_init=(1, 3, 256, 256),
    seed=42,
    get_trajectory=True,
)
dinv.utils.plot(
    sample,
    titles="Unconditional generation",
    figsize=(figsize, figsize),
)


# %% Posterior sampling
# ---------------------
#
# Similar to other denoisers in DeepInv, the wrapped diffusers model can be used for posterior sampling.
# Below we use VP-SDE for posterior sampling in an inpainting problem.

# Initialize the physics and the VP-SDE

mask = torch.ones_like(x)
mask[..., 70:150, 120:180] = 0
physics = dinv.physics.Inpainting(
    mask=mask,
    img_size=x.shape[1:],
    device=device,
    noise_model=dinv.physics.GaussianNoise(0.05),
)

y = physics(x)
sde = VariancePreservingDiffusion(device=device, dtype=dtype)

# %% Define the posterior sampler with a noisy data-fidelity term

from deepinv.sampling import DPSDataFidelity

model = PosteriorDiffusion(
    data_fidelity=DPSDataFidelity(denoiser=denoiser, weight=0.3),
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
)

# %% Run posterior sampling
posterior_sample = model(
    y=y,
    physics=physics,
    x_init=(1, 3, 256, 256),
    seed=15,
)
dinv.utils.plot(
    [x, y, posterior_sample],
    titles=["Original image", "Measurement", "Posterior sample"],
    figsize=(figsize * 3, figsize),
)
