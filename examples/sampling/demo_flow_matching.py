r"""
Flow-Matching and closed-form MMSE denoiser
==============================================

This demo shows you how to use Flow-Matching to perform unconditional image generation or posterior sampling.
In particular, in this demo, we will show how to perform generation using the closed-form MMSE denoiser which is calculated from a given dataset of clean images.
This is equivalent to flow-matching with closed-form velocity, analyzed for example in :footcite:`bertrand2025closed`.
"""

# %%
import torch
import deepinv as dinv
from deepinv.sampling import (
    PosteriorDiffusion,
    DPSDataFidelity,
    EulerSolver,
    FlowMatching,
)
from torchvision import datasets, transforms
from deepinv.models import MMSE

# %% Define the MMSE denoiser
# ----------------------------------------------------------------
#
# The closed-form MMSE denoser is calculated by computing the distance between the input image and all the points of the dataset.
# This can be quite long to compute for large images and large datasets.
# In this toy example, we use the training set of MNIST.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else 8
dtype = torch.float64
figsize = 2.5

# We use the closed-form MMSE denoiser defined using as atoms the testset of MNIST.
# The deepinv MMSE denoiser takes as input a dataloader.
dataset = datasets.MNIST(
    root=".", train=True, download=True, transform=transforms.ToTensor()
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=512, shuffle=False, num_workers=num_workers
)
# Since the MNIST dataset is relatively small, we can also load it entirely in memory as a tensor
tensors = torch.cat([data[0] for data in iter(dataloader)]).to(device)
denoiser = MMSE(dataloader=tensors.clone(), device=device, dtype=torch.float32)


# %% Define the Flow-Matching ODE and perform unconditional generation
# ----------------------------------------------------------------
# The FlowMatching module takes as input the denoiser and the ODE solver.
#

num_steps = 100

rng = torch.Generator(device).manual_seed(42)
timesteps = torch.linspace(0.99, 0.0, num_steps)
solver = EulerSolver(timesteps=timesteps, rng=rng)
sde = FlowMatching(denoiser=denoiser, solver=solver, device=device)

sample, trajectory = sde(
    x_init=(1, 1, 28, 28),
    seed=42,
    get_trajectory=True,
)

dinv.utils.plot(
    sample,
    titles="Unconditional FM generation",
    save_fn="FM_sample.png",
    figsize=(figsize, figsize),
)

# dinv.utils.save_videos(
#     trajectory.cpu(),
#     time_dim=0,
#     titles=["FM Trajectory"],
#     save_fn="FM_trajectory.gif",
#     figsize=(figsize, figsize),
# )


# %% Perform posterior sampling
# ----------------------------------------------------------------
# Now, we can use the Flow-Matching model to perform posterior sampling.
# When the data fidelity is given, together with the measurements and the physics, we can perform posterior sampling for inverse problems.
# For example, consider the inpainting problem, where we have a noisy image and we want to recover the original image.
# We can use the :class:`deepinv.sampling.DPSDataFidelity` as the data fidelity term.

x = tensors[0:1]
mask = torch.ones_like(x)
mask[..., 10:20, 10:20] = 0.0
physics = dinv.physics.Inpainting(
    img_size=x.shape[1:],
    mask=mask,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0.1),
)
y = physics(x)

weight = 1e-2  # guidance strength
dps_fidelity = DPSDataFidelity(denoiser=denoiser, weight=weight)


model = PosteriorDiffusion(
    data_fidelity=dps_fidelity,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
)
# To perform posterior sampling, we need to provide the measurements, the physics and the solver.
# Moreover, when the physics is given, the initial point can be inferred from the physics if not given explicitly.
seed_1 = 1
x_hat, trajectory = model(
    y,
    physics,
    x_init=None,
    seed=seed_1,
    get_trajectory=True,
)

# Here, we plot the original image, the measurement and the posterior sample
dinv.utils.plot(
    [x, y, x_hat],
    show=True,
    titles=["Original", "Measurement", "Posterior sample"],
    figsize=(figsize * 3, figsize),
    save_fn="FM_posterior.png",
)
# We can also save the trajectory of the posterior sample
# dinv.utils.save_videos(
#     trajectory,
#     time_dim=0,
#     titles=["Posterior sample with FM"],
#     save_fn="FM_posterior_trajectory.gif",
#     figsize=(figsize, figsize),
# )

# %%
