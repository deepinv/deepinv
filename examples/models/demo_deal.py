"""
DEAL reconstruction demo.
====================================================================================================

This example shows how to use the Deep Equilibrium Attention Least Squares
(DEAL) reconstruction model in DeepInverse for a simple deblurring problem.

DEAL solves linear inverse problems using a learned equilibrium-based
regularizer combined with conjugate gradient iterations. It can be used for
image restoration and reconstruction tasks such as denoising, deblurring,
and computed tomography reconstruction.

Given measurements :math:`y = Ax + n`, DEAL computes a reconstruction by
combining a learned spatially-varying regularizer with the solution of
a linear least-squares subproblem. This implementation is adapted from the
official DEAL repository:
https://github.com/mehrsapo/DEAL
The DEAL algorithm solves the following optimization problem:

.. math::

    \hat{x} = \arg\min_x \frac{1}{2}\|Ax - y\|^2 + \lambda R_\theta(x),

where :math:`R_\theta(x)` is a learned spatially adaptive regularizer.

The reconstruction is obtained iteratively using a fixed-point scheme.
At each iteration, a linearized least-squares subproblem is solved using
conjugate gradient.
"""

# %%
# Import packages and load a grayscale example image from Set3C.

import torch

from deepinv.loss.metric import PSNR
from deepinv.models import DEAL
from deepinv.physics import Blur, GaussianNoise
from deepinv.physics.blur import gaussian_blur
from deepinv.utils import load_example, plot

device = "cuda" if torch.cuda.is_available() else "cpu"

# We use the grayscale DEAL checkpoint in this demo, so we keep one channel.
x = load_example("butterfly.png", img_size=128).to(device)[:, 0:1, :, :]

# %%
# Define the blur + noise forward model and generate the measurement.

noise_std = 0.01
physics = Blur(
    filter=gaussian_blur(sigma=(2.0, 2.0), angle=0.0),
    noise_model=GaussianNoise(sigma=noise_std),
    padding="circular",
    device=device,
)

y = physics(x)

# %%
# Load pretrained DEAL weights from the original repository.
#
# Both grayscale and color pretrained checkpoints are available. Here we use
# the grayscale model with color=False.

model = DEAL(
    pretrained="download",
    sigma=25.0,
    lam=10.0,
    max_iter=10,
    auto_scale=False,
    color=False,
    device=device,
    clamp_output=True,
)

# %%
# Reconstruct the image, compare with a linear baseline, and display PSNR.

with torch.no_grad():
    x_lin = physics.A_dagger(y)
    x_hat = model(y, physics)
psnr = PSNR()
psnr_y = psnr(y, x).item()
psnr_lin = psnr(x_lin, x).item()
psnr_hat = psnr(x_hat, x).item()

plot(
    [x, y, x_lin, x_hat],
    titles=[
        "Ground truth",
        f"Blurred measurement",
        f"Linear reconstruction",
        f"DEAL reconstruction",
    ],
    subtitles=[
        "PSNR:",
        f"{psnr_y:.2f} dB",
        f"{psnr_lin:.2f} dB",
        f"{psnr_hat:.2f} dB",
    ],
    figsize=(10, 3),
)
