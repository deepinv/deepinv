r"""
PIDAL + TV prior for Deconvolution
======================================

Demonstrates using the PIDAL (see :footcite:t:`figueiredo_restoration_2010`) scheme with a total-variation (TV) prior
for Poisson decopnvolution on a blurred image.

This method is an alternative to ADMM for non-denoising Poisson inverse problems as it provides a splitting procedure.
"""

# %%
import deepinv as dinv
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Load butterfly image
#
img_size = (128, 128)
x = dinv.utils.load_example("butterfly.png", img_size=img_size, device=device, grayscale=False)

# %%
# Create Blur physics and simulate measurements
#
sigma = 1.5
ksize = 6 * int(sigma) + 1
kernel = dinv.physics.functional.gaussian_blur(
    psf_size=(ksize, ksize),
    sigma=sigma,
    device=device
)
#
gain = 0.1
noise_model = dinv.physics.PoissonNoise(gain=gain, normalize=True)
physics = dinv.physics.Blur(
    filter=kernel,
    padding="valid",
    device=device
)
#
y = physics(x)
y = noise_model(y)
#
y_size = y.shape[-2:]
x_cropped = x[..., (x.shape[-2] - y_size[-2]) // 2:-(x.shape[-2] - y_size[-2]) // 2, (x.shape[-1] - y_size[-1]) // 2:-(x.shape[-1] - y_size[-1]) // 2]
psnr = dinv.metric.PSNR().forward(y, x_cropped).item()
#
dinv.utils.plot(
    [x, y],
    titles=["Ground truth", "Blurred image"],
    subtitles=["", f"PSNR: {psnr:.2f} dB"],
    figsize=(8, 4)
)

# %%
# Define PIDAL optimizer with TV prior and Poisson likelihood data fidelity
prior = dinv.optim.prior.TVPrior(
    n_it_max=100
)
data_fidelity = dinv.optim.PoissonLikelihood(
    gain=gain,
    denormalize=True
)
model = dinv.optim.PIDAL(
    data_fidelity=data_fidelity,
    prior=prior,
    max_iter=100,
    stepsize=1.0,
    lambda_reg=2.0
)

# %%
# Run PIDAL reconstruction
x_hat = model(y, physics)
#
psnr  = dinv.metric.PSNR().forward(x_hat, x).item()
dinv.utils.plot(
    [x, x_hat],
    titles=["Ground truth", "PIDAL reconstruction"],
    subtitles=["", f"PSNR: {psnr:.2f} dB"],
    figsize=(8, 4)
)