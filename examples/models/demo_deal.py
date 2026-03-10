"""
DEAL reconstruction demo.

This example shows how to use the DEAL reconstruction model in DeepInverse
for a simple deblurring problem.
"""

import torch

import deepinv as dinv
from deepinv.models import DEAL
from deepinv.physics import Blur, GaussianNoise
from deepinv.physics.blur import gaussian_blur
from deepinv.utils import load_example, plot


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a color example image
x = load_example("celeba_example.jpg", img_size=128).to(device)

# Define blur + Gaussian noise physics
noise_std = 0.01
physics = Blur(
    filter=gaussian_blur(sigma=(2.0, 2.0), angle=0.0),
    noise_model=GaussianNoise(sigma=noise_std),
    device=device,
)

# Generate measurement
y = physics(x)

# Load DEAL model
model = DEAL(
    pretrained="download",
    sigma=25.0,
    lam=10.0,
    max_iter=10,
    auto_scale=False,
    color=True,
    device=device,
    clamp_output=True,
)

# Run reconstruction
with torch.no_grad():
    x_hat = model(y, physics)

# Display results
plot(
    [x, y, x_hat],
    titles=["Ground truth", "Blurred measurement", "DEAL reconstruction"],
)
