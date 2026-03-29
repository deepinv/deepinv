"""
DEAL denoising demo.
====================================================================================================

This example shows how to use the Deep Equilibrium Attention Least Squares
(DEAL) reconstruction model in DeepInverse for grayscale image denoising.

DEAL solves linear inverse problems using a learned equilibrium-based
regularizer combined with iterative least-squares updates. It can be used for
image restoration and reconstruction tasks such as denoising, deblurring,
and computed tomography reconstruction.

The pretrained checkpoints used in this integration are trained for denoising.
This demo therefore illustrates the pretrained model in its native setting.

Given measurements :math:`y = x + n`, DEAL computes a reconstruction by
combining a learned spatially-varying regularizer with the solution of
a regularized least-squares subproblem. This implementation is adapted from the
official `DEAL repository <https://github.com/mehrsapo/DEAL>`_.

The DEAL algorithm solves the following optimization problem:

.. math::

    \hat{x} = \arg\min_x \frac{1}{2}\|x - y\|^2 + \lambda g_\theta(x),

where :math:`g_\theta(x)` is a learned spatially adaptive regularizer.

The reconstruction is obtained iteratively using a fixed-point scheme.
At each iteration, a linearized least-squares subproblem is solved using
conjugate gradient.
"""

# %%
# Import packages and load a grayscale example image from Set3C.

import torch

from deepinv.loss.metric import PSNR
from deepinv.models import DEAL, DEALRegularizer
from deepinv.physics import Denoising, GaussianNoise
from deepinv.utils import load_example, plot

device = "cuda" if torch.cuda.is_available() else "cpu"

# We use the grayscale DEAL checkpoint in this demo, so we keep one channel.
x = load_example("butterfly.png", img_size=128).to(device)[:, 0:1, :, :]

# %%
# Define the denoising forward model and generate the measurement.

sigma255 = 25.0
noise_std = sigma255 / 255.0

physics = Denoising(GaussianNoise(sigma=noise_std)).to(device)

y = physics(x)

# %%
# Load pretrained DEAL weights from the original repository.
#
# Both grayscale and color pretrained checkpoints are available. Here we use
# the grayscale model with color=False.

model = DEAL(
    pretrained="download",
    sigma=sigma255,
    lam=10.0,
    max_iter=10,
    auto_scale=False,
    color=False,
    device=device,
    clamp_output=True,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"DEAL number of parameters: {n_params:,}")

prior = DEALRegularizer(model.model)

with torch.no_grad():
    grad_prior = prior.grad(x, sigma=sigma255)
    mask = model.model.mask.mean(dim=1, keepdim=True)

print(f"Standalone DEAL prior gradient shape: {tuple(grad_prior.shape)}")

# %%
# Reconstruct the image, compare with a baseline, and display PSNR.

with torch.no_grad():
    x_lin = y.clone()
    x_hat = model(y, physics)

psnr = PSNR()
psnr_y = psnr(y, x).item()
psnr_lin = psnr(x_lin, x).item()
psnr_hat = psnr(x_hat, x).item()

print(f"PSNR noisy: {psnr_y:.2f} dB")
print(f"PSNR linear: {psnr_lin:.2f} dB")
print(f"PSNR DEAL: {psnr_hat:.2f} dB")

plot(
    [x, y, x_lin, x_hat, mask],
    titles=[
        "Ground truth",
        "Noisy measurement",
        "Identity baseline",
        "DEAL reconstruction",
        "DEAL mask",
    ],
    subtitles=[
        "",
        f"PSNR: {psnr_y:.2f} dB",
        f"PSNR: {psnr_lin:.2f} dB",
        f"PSNR: {psnr_hat:.2f} dB",
        "Mean over channels",
    ],
    figsize=(13, 3),
)
