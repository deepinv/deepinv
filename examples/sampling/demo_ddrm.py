r"""
Image reconstruction with a diffusion model
====================================================================================================

This code shows you how to use the DDRM diffusion algorithm :footcite:t:`kawar2022denoising` to reconstruct images and also compute the
uncertainty of a reconstruction from incomplete and noisy measurements.

The DDRM method requires that:

* The operator has a singular value decomposition (i.e., the operator is a :class:`deepinv.physics.DecomposablePhysics`).
* The noise is Gaussian with known standard deviation (i.e., the noise model is :class:`deepinv.physics.GaussianNoise`).
"""

# %%
import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
import numpy as np
from deepinv.utils.demo import load_example

# %%
# Load example image from the internet
# --------------------------------------------------------------
#
# This example uses an image of Messi.

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

x = load_example("messi.jpg", img_size=32).to(device)


# %%
# Define forward operator and noise model
# --------------------------------------------------------------
#
# We use image inpainting as the forward operator and Gaussian noise as the noise model.

sigma = 0.1  # noise level
physics = dinv.physics.Inpainting(
    mask=0.5,
    img_size=x.shape[1:],
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
)


# %%
# Define the MMSE denoiser
# --------------------------------------------------------------
#
# The diffusion method requires an MMSE denoiser that can be evaluated a various noise levels.
# Here we use a pretrained DRUNET denoiser from the :ref:`denoisers <denoisers>` module.

denoiser = dinv.models.DRUNet(pretrained="download").to(device)

# %%
# Create the Monte Carlo sampler
# --------------------------------------------------------------
#
# We can now reconstruct a noisy measurement using the diffusion method.
# We use the DDRM method from :class:`deepinv.sampling.DDRM`, which works with inverse problems that
# have a closed form singular value decomposition of the forward operator.
# The diffusion method requires a schedule of noise levels ``sigmas`` that are used to evaluate the denoiser.

sigmas = np.linspace(1, 0, 100) if torch.cuda.is_available() else np.linspace(1, 0, 10)

diff = dinv.sampling.DDRM(denoiser=denoiser, etab=1.0, sigmas=sigmas, verbose=True)

# %%
# Generate the measurement
# ---------------------------------------------------------------------------------
# We apply the forward model to generate the noisy measurement.

y = physics(x)

# %%
# Run the diffusion algorithm and plot results
# ---------------------------------------------------------------------------------
# The diffusion algorithm returns a sample from the posterior distribution.
# We compare the posterior mean with a simple linear reconstruction.

xhat = diff(y, physics)

# compute linear inverse
x_lin = physics.A_adjoint(y)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")
print(f"Diffusion PSNR: {dinv.metric.PSNR()(x, xhat).item():.2f} dB")

# plot results
error = (xhat - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
imgs = [x_lin, x, xhat]
plot(imgs, titles=["measurement", "ground truth", "DDRM reconstruction"])

# %%
# Create a Monte Carlo sampler
# ---------------------------------------------------------------------------------
# Running the diffusion gives a single sample of the posterior distribution.
# In order to compute the posterior mean and variance, we can use multiple samples.
# This can be done using the :class:`deepinv.sampling.DiffusionSampler` class, which converts
# the diffusion algorithm into a fully fledged Monte Carlo sampler.
# We set the maximum number of iterations to 10, which means that the sampler will run the diffusion 10 times.

f = dinv.sampling.DiffusionSampler(diff, max_iter=10)


# %%
# Run sampling algorithm and plot results
# ---------------------------------------------------------------------------------
# The sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean with a simple linear reconstruction.

mean, var = f(y, physics)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")
print(f"Posterior mean PSNR: {dinv.metric.PSNR()(x, mean).item():.2f} dB")

# plot results
error = (mean - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [
    x_lin,
    x,
    mean,
    std / std.flatten().max(),
    error / error.flatten().max(),
]
plot(
    imgs,
    titles=[
        "measurement",
        "ground truth",
        "post. mean",
        "post. std",
        "abs. error",
    ],
)

# %%
# :References:
#
# .. footbibliography::
