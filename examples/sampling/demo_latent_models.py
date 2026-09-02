# %%
import torch
import deepinv as dinv
from deepinv.models.wrapper import DiffusersDenoiserWrapper

device = dinv.utils.get_device()
dtype = torch.float32
figsize = 2.5

from deepinv.sampling import PosteriorDiffusion, EulerSolver, VariancePreservingDiffusion, VarianceExplodingDiffusion
from deepinv.optim import ZeroFidelity


# %%
# ----------------------------------------------------
#
# Let us first load a pretrained latent diffusion model from the HuggingFace Hub. Here, we use the `runwayml/stable-diffusion-v1-5` model.
# This model is trained on 512x512 images.

# We can wrap any diffusers latent model as a DeepInv denoiser using one line of code:
denoiser = DiffusersDenoiserWrapper(
    model_id="runwayml/stable-diffusion-v1-5", pipeline_name="DiffusionPipeline", device=device, clip_output=False,
)
from diffusers import DDIMScheduler
denoiser.scheduler = DDIMScheduler.from_config(denoiser.scheduler.config)

# Load an example image
x = dinv.utils.load_example(
    "celeba_example2.jpg",
    img_size=512,
    resize_mode="resize",
).to(device)

# Define the prompt
prompt = ["a high resolution photo of a cat on a grass field, 4K, sharp"]


# %%
# ---------------------------------
#
# It is also possible to use the wrapped model for unconditional image generation.
# The model was trained with DDPM scheduler, however we can use it with any SDE provided in DeepInv.
# Here, we use the Variance Preserving SDE with Euler solver for sampling.

num_steps = 200
timesteps = torch.linspace(1, 0.001, num_steps)
rng = torch.Generator(device)
solver = EulerSolver(timesteps=timesteps, rng=rng)

sde = VariancePreservingDiffusion(
    beta_min=0.85,
    beta_max=12.0,
    scaled_linear=True,
    device=device,
    dtype=dtype,
    alpha=0.5,
)

model = PosteriorDiffusion(
    data_fidelity=ZeroFidelity(),
    sde=sde,
    denoiser=denoiser,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
    minus_one_one=False,
)

z = torch.randn(
    1, 4, 64, 64,
    device=device,
    dtype=dtype,
)

sample, trajectory = model(
    y=None,
    physics=None,
    x_init=z,
    seed=42,
    get_trajectory=True,
    prompt=prompt,
    input_in_minus_one_one=True,   # important
    denoise_output=False,
    guidance_scale = 6,
)
dinv.utils.plot(
    sample,
    titles="Unconditional generation",
    figsize=(figsize, figsize),
)


# %%
# ---------------------
#
# Similar to other denoisers in DeepInv, the wrapped diffusers model can be used for posterior sampling.
# Below we use the same VP-SDE for posterior sampling in an inpainting problem.

# Initialize the physics

mask = torch.ones_like(x)
mask[..., 128:384, 128:384] = 0
physics = dinv.physics.Inpainting(
    mask=mask,
    img_size=x.shape[1:],
    device=device,
    noise_model=dinv.physics.GaussianNoise(0.01),
)

y = physics(x)


# %%
# We first run LDPS to show how the posterior sampling can fail in certain scenarios.

from deepinv.sampling import DPSDataFidelity

num_steps = 500
rng = torch.Generator(device)
solver = EulerSolver(timesteps=timesteps, rng=rng)

model = PosteriorDiffusion(
    data_fidelity=DPSDataFidelity(denoiser=denoiser, sde=sde, timesteps=timesteps, original_algo=True, weight=1.0),
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
    minus_one_one=False,
)


# %%
# Define the prompt, which by default is set to be the null prompt
prompt = [""]

posterior_sample = model(
    y=y,
    physics=physics,
    x_init=z,
    seed=15,
    prompt=prompt,
    input_in_minus_one_one=True,   # important
    denoise_output=False,
    guidance_scale = 1,
)
dinv.utils.plot(
    [x, y, posterior_sample],
    titles=["Original image", "Measurement", "Posterior sample"],
    figsize=(figsize * 3, figsize),
)


# %%
# Next, we demonstrate posterior sampling using the PSLD data fidelity.

from deepinv.sampling import PSLDDataFidelity

num_steps = 500
rng = torch.Generator(device)
solver = EulerSolver(timesteps=timesteps, rng=rng)

model = PosteriorDiffusion(
    data_fidelity=PSLDDataFidelity(denoiser=denoiser, sde=sde, timesteps=timesteps, omega=1.0, gamma=0.1),
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
    minus_one_one=False,
)


# %%
# Define the prompt, which by default is set to be the null prompt.
prompt = [""]

posterior_sample = model(
    y=y,
    physics=physics,
    x_init=z,
    seed=15,
    prompt=prompt,
    input_in_minus_one_one=True,   # important
    denoise_output=False,
    guidance_scale = 1,
)
dinv.utils.plot(
    [x, y, posterior_sample],
    titles=["Original image", "Measurement", "Posterior sample"],
    figsize=(figsize * 3, figsize),
)