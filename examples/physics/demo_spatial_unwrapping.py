r"""
Spatial Unwrapping Demo
======================

This demo shows the use of the :class:`deepinv.physics.SpatialUnwrapping` forward model and the Itoh data-fidelity for phase unwrapping problems.
It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.

Sections:
    1. Imports and setup
    2. Load image and preprocess
    3. Apply blur
    4. Wrap phase and add noise
    5. Invert with DCT and ADMM (ItohFidelity)
    6. Visualize results
"""

# %%
# Imports and setup
# ------------------------------------
import numpy as np
import torch
from deepinv.utils.plotting import plot

np.random.seed(0)
torch.manual_seed(0)
from deepinv.physics.spatial_unwrapping import SpatialUnwrapping
import deepinv as dinv
from deepinv.utils.demo import load_example
import torchvision.transforms as transforms


def channel_norm(x):
    x = x - x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    x = x / x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    return x


# %%
# Load image and preprocess
# ------------------------------------
# Load example image from deepinv and preprocess
size = 256
dr = 2  # dynamic range
dtype = torch.float32
device = "cpu"
img_size = (size, size)
mode = "floor"  # available modes: "round", "floor"


x_rgb = load_example(
    "CBSD_0010.png", grayscale=False, device=device, dtype=dtype, img_size=img_size
)
x_rgb = channel_norm(x_rgb) * dr

factor = 3
resize = transforms.Resize(size=(img_size[0] * factor, img_size[1] * factor))
x_rgb = resize(x_rgb)

if mode == "round":
    x_rgb = x_rgb - dr / 2

# %%
# Apply blur
# ------------------------------------
# We apply a slight Gaussian blur to the image to simulate the low bandwidth assumption of the phase map.
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(1, 1), angle=0.0)
blur_op = dinv.physics.Blur(filter_0, device=device)
x_rgb = blur_op(x_rgb)


# %%
# Add Gaussian noise and wrap phase
# ------------------------------------
# Include Gaussian noise and wrap phase using SpatialUnwrapping physics
noise_model = dinv.physics.GaussianNoise(sigma=0.1)
physics = SpatialUnwrapping(threshold=1.0, mode=mode, noise_model=noise_model)
phase_map = x_rgb
wrapped_phase = physics(phase_map)

# %%
# Invert with DCT and ADMM (ItohFidelity)
# ------------------------------------
# We provide two inversion methods: a simple DCT-based inversion and an ADMM-based inversion using the Itoh fidelity term and TV prior.


# DCT-based inversion
x_est = physics.A_dagger(wrapped_phase)

# ADMM-based inversion with TV prior and Itoh fidelity
stepsize = 1e-4
lam = 2.0 / stepsize
prior = dinv.optim.TVPrior(n_it_max=10)
fidelity = dinv.optim.ItohFidelity()
params_algo = {"stepsize": stepsize, "lambda": lam, "g_param": 1.0}
model = dinv.optim.optim_builder(
    iteration="ADMM",
    prior=prior,
    data_fidelity=fidelity,
    max_iter=10,
    verbose=False,
    params_algo=params_algo,
)
x_model = model(wrapped_phase, physics, compute_metrics=False)


# %%
# Visualize results
# ------------------------------------
psnr_fn = dinv.metric.PSNR()
ssim_fn = dinv.metric.SSIM()

# Normalize for display
x_est = channel_norm(x_est)
x_model = channel_norm(x_model)
phase_map = channel_norm(phase_map)

# Compute metrics
psnr_admm = psnr_fn(phase_map, x_model).item()
psnr_dct = psnr_fn(phase_map, x_est).item()
ssim_admm = ssim_fn(phase_map, x_model).item()
ssim_dct = ssim_fn(phase_map, x_est).item()

# Plot results
imgs = [wrapped_phase[0], phase_map[0], x_est[0], x_model[0]]
titles = [
    "Wrapped Phase",
    "Original Phase",
    f"DCT Inversion\n PSNR={psnr_dct:.2f} SSIM={ssim_dct:.2f}",
    f"ADMM Inversion\n PSNR={psnr_admm:.2f} SSIM={ssim_admm:.2f}",
]
plot(imgs, titles=titles, cmap="gray", figsize=(20, 10))
