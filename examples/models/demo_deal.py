"""
DEAL denoising and reconstruction
====================================================================================================

This example shows how to use the Deep Equilibrium Attention Least Squares
(DEAL) model in DeepInverse for both denoising and a simple reconstruction
setting.

The pretrained DEAL model is primarily designed for denoising (with noise
levels expressed in the [0,255] scale). We therefore demonstrate:

1) Denoising (native setting of the pretrained model)
2) Reconstruction (inpainting) using the same model

DEAL solves inverse problems by combining a learned spatially adaptive
regularizer with iterative least-squares updates.

This implementation is adapted from the official
`DEAL repository <https://github.com/mehrsapo/DEAL>`_.
"""

# %%
# Import packages and load a grayscale example image.

import torch

from deepinv.loss.metric import PSNR
from deepinv.models import DEAL, DEALRegularizer
from deepinv.physics import Denoising, GaussianNoise, Inpainting
from deepinv.utils import load_example, plot

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Load image (grayscale)
x = load_example("butterfly.png", img_size=128, device=device, grayscale=True)

# %%
# IMPORTANT: sigma handling
# physics uses [0,1], model uses [0,255]

sigma255 = 25.0
sigma01 = sigma255 / 255.0

# %%
# Load pretrained DEAL model

model = DEAL(
    pretrained="download",
    sigma=sigma255,  # IMPORTANT: must be in [0,255]
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
psnr = PSNR()

# %%
# ------------------------------------------------------------
# 1) DENOISING
# ------------------------------------------------------------

physics_denoise = Denoising(GaussianNoise(sigma=sigma01)).to(device)
y_denoise = physics_denoise(x)

with torch.no_grad():
    grad_prior = prior.grad(x, sigma=sigma255)
    x_hat_denoise = model(y_denoise, physics_denoise)
    mask_denoise = model.model.mask.mean(dim=1, keepdim=True)

print(f"Standalone DEAL prior gradient shape: {tuple(grad_prior.shape)}")

psnr_noisy = psnr(y_denoise, x).item()
psnr_denoise = psnr(x_hat_denoise, x).item()

print(f"[Denoising] PSNR noisy: {psnr_noisy:.2f} dB")
print(f"[Denoising] PSNR DEAL: {psnr_denoise:.2f} dB")

# The mask corresponds to the spatially varying weights learned by DEAL.
# It controls how strongly the regularizer acts at each pixel.
plot(
    [x, y_denoise, x_hat_denoise, mask_denoise],
    titles=[
        "Ground truth",
        "Noisy input",
        "DEAL denoising",
        "DEAL mask",
    ],
    subtitles=[
        "",
        f"PSNR: {psnr_noisy:.2f} dB",
        f"PSNR: {psnr_denoise:.2f} dB",
        "Mean over channels",
    ],
    figsize=(11, 3),
)

# %%
# ------------------------------------------------------------
# 2) RECONSTRUCTION: INPAINTING
# ------------------------------------------------------------

# Create random mask (50% missing pixels)
mask = (torch.rand(1, 1, 128, 128, device=device) > 0.5).float()

physics_inpaint = Inpainting(
    img_size=(1, 128, 128),
    mask=mask,
    noise_model=GaussianNoise(sigma=0.0),
).to(device)

y_inpaint = physics_inpaint(x)

with torch.no_grad():
    x_lin = physics_inpaint.A_adjoint(y_inpaint)
    x_hat_inpaint = model(y_inpaint, physics_inpaint)
    mask_inpaint = model.model.mask.mean(dim=1, keepdim=True)

psnr_meas_inpaint = psnr(y_inpaint, x).item()
psnr_lin_inpaint = psnr(x_lin, x).item()
psnr_deal_inpaint = psnr(x_hat_inpaint, x).item()

print(f"[Inpainting] PSNR measurement: {psnr_meas_inpaint:.2f} dB")
print(f"[Inpainting] PSNR linear: {psnr_lin_inpaint:.2f} dB")
print(f"[Inpainting] PSNR DEAL: {psnr_deal_inpaint:.2f} dB")

# Again, the mask shows the learned spatial weighting of the regularizer.
plot(
    [x, y_inpaint, x_lin, x_hat_inpaint, mask_inpaint],
    titles=[
        "Ground truth",
        "Masked measurement",
        "Adjoint baseline",
        "DEAL reconstruction",
        "DEAL mask",
    ],
    subtitles=[
        "",
        f"PSNR: {psnr_meas_inpaint:.2f} dB",
        f"PSNR: {psnr_lin_inpaint:.2f} dB",
        f"PSNR: {psnr_deal_inpaint:.2f} dB",
        "Mean over channels",
    ],
    figsize=(13, 3),
)
