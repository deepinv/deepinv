# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 15:18:28 2025

@author: olehl
"""

import deepinv as dinv
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP, Zero
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.demo import load_example
from deepinv.utils.plotting import plot
from deepinv.optim.phase_retrieval import (
    correct_global_phase,
    cosine_similarity,
    spectral_methods,
)
from deepinv.models.complex import to_complex_denoiser

BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"
# Set global random seed to ensure reproducibility.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Image size
img_size = 32
# The pixel values of the image are in the range [0, 1].
x = load_example(
    "SheppLogan.png",
    img_size=img_size,
    grayscale=True,
    resize_mode="resize",
    device=device,
)
print(x.min(), x.max())

plot(x, titles="Original image")

x_phase = torch.exp(1j * x * torch.pi - 0.5j * torch.pi)
x_real = x

# Every element of the signal should have unit norm.
assert torch.allclose(x_phase.real**2 + x_phase.imag**2, torch.tensor(1.0))

# Define physics information
oversampling_ratio = 5.0
img_shape = x.shape[1:]
m = int(oversampling_ratio * torch.prod(torch.tensor(img_shape)))
n_channels = 1  # 3 for color images, 1 for gray-scale images

# Create the physics
# dtype=torch.float to make the problem real
physics = dinv.physics.RandomPhaseRetrieval(
    m=m,
    img_shape=img_shape,
    dtype=torch.float,
    device=device,
)

# Generate measurements
y = physics(x_real)

# Spectral methods return a tensor with unit norm.
x_phase_spec = physics.A_dagger(y, n_iter=300)

# correct possible global phase shifts
# x_spec = correct_global_phase(x_phase_spec, x_real)
# extract phase information and normalize to the range [0, 1]
# x_spec = torch.angle(x_spec) / torch.pi + 0.5
plot([x, x_phase_spec], titles=["Signal", "Reconstruction"], rescale_mode="clip")


physics_c = dinv.physics.RandomPhaseRetrieval(
    m=m,
    img_shape=img_shape,
    # dtype=torch.float,
    device=device,
)

# Generate measurements
y_c = physics_c(x_phase)

# Spectral methods return a tensor with unit norm.
x_phase_spec = physics_c.A_dagger(y_c, n_iter=300)

# correct possible global phase shifts
x_spec = correct_global_phase(x_phase_spec, x_phase)
# extract phase information and normalize to the range [0, 1]
x_spec = torch.angle(x_spec) / torch.pi + 0.5
plot([x, x_spec], titles=["Signal", "Reconstruction"], rescale_mode="clip")