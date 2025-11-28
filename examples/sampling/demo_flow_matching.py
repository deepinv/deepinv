r"""
Flow-Matching and closed-form MMSE denoiser
==============================================

This demo shows you how to use Flow-Matching to perform unconditional image generation or posterior sampling.
In particulat, in this demo, we will whow how to perform generation using the closed-form MMSE denoiser which is calculated from a given dataset of clean images.
This is equivalent to flow-matching with closed-form velocity, analyzed for example in :footcite:`bertrand2025closed`.
"""

import torch
import deepinv as dinv
from deepinv.sampling import (
    VarianceExplodingDiffusion,
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
# In this toy example, we use as dataset the testset of MNIST. 
#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
figsize = 2.5

# We use the closed-form MMSE denoiser defined using as atoms the testset of MNIST.
# The deepinv MMSE denoiser takes as input a dataloader.
dataset = datasets.MNIST(root='.', train=False, download=True, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, num_workers=4)
denoiser = dinv.models.MMSE(dataloader=dataloader, device=device, dtype=torch.float32)


# %% Define the Flow-Matching ODE 
# ----------------------------------------------------------------
# The FlowMatching module takes as input the denoiser and the ODE solver.
# 

num_steps = 10

# With FM
rng = torch.Generator(device).manual_seed(42)
timesteps = torch.linspace(0.99, 0., num_steps)
solver = EulerSolver(timesteps=timesteps, rng=rng)
model = FlowMatching(
    denoiser=denoiser,
    solver=solver,
    device=device
)

sample, trajectory = model(
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

dinv.utils.save_videos(
    trajectory.cpu()[::1],
    time_dim=0,
    titles=["FM Trajectory"],
    save_fn="FM_trajectory.gif",
    figsize=(figsize, figsize),
)



# With score model
from deepinv.models import NCSNpp
denoiser = NCSNpp(pretrained="download").to(device)
