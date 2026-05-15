r"""
DEAL denoising and reconstruction
====================================================================================================

This example shows how to use the Deep Equilibrium Attention Least Squares
(DEAL) model in DeepInverse for both denoising and a simple reconstruction
setting.

The reconstruction is obtained by solving

.. math::

    \hat{x} = \arg\min_x \frac{1}{2}\|Ax - y\|^2 + \lambda g_{\theta}(x),

where :math:`A` is the forward operator, :math:`y` are the measurements, and
:math:`g_{\theta}` is the learned adaptive regularizer.

In DEAL, the regularizer is induced by a masked linear operator

.. math::

    L_{\theta,c}(u, x) = m_{\theta,c}(u) \odot K_{\theta,c}x,

which gives

.. math::

    g_{\theta}(u, x)
    =
    \sum_{c=1}^{C}
    \frac{1}{2}
    \|m_{\theta,c}(u) \odot K_{\theta,c}x\|_2^2.

The fixed-point iterations used by the solver are

.. math::

    x^{(k+1)}
    \approx
    \arg\min_x
    \frac{1}{2}\|Ax-y\|^2
    +
    \lambda
    \nabla_x g_{\theta}(u=x^{(k)}, x)^\top x.

Each subproblem is solved approximately with conjugate gradient.

DEAL solves inverse problems by minimizing a data-fidelity term together with
a learned adaptive regularizer. In the implementation, this regularizer is
induced by a masked linear operator, where learned filters are modulated by
spatially varying masks predicted by the network. The reconstruction is then
refined through iterative least-squares updates, where the regularizer is
recomputed from the current iterate.

This implementation is adapted from the official
`DEAL repository <https://github.com/mehrsapo/DEAL>`_.

Here, the model is illustrated first for Gaussian denoising, and then for a
simple inpainting reconstruction problem.
"""

# %%
# Import packages and load a grayscale example image.

import torch

from deepinv.loss.metric import PSNR
from deepinv.models import DEAL
from deepinv.physics import Denoising, GaussianNoise, Inpainting
from deepinv.utils import load_example, plot

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Load image (grayscale)
x = load_example("butterfly.png", img_size=128, device=device, grayscale=True)

# %%
# Noise level in the normalized [0,1] convention used by DeepInverse.

sigma = 0.1

# %%
# Load pretrained DEAL model

model = DEAL(
    pretrained="download",
    sigma_denoiser=sigma,
    lambda_reg=10.0,
    max_iter=10,
    auto_scale=False,
    color=False,
    device=device,
    clamp_output=True,
)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"DEAL number of parameters: {n_params:,}")

psnr = PSNR()

# %%
# Denoising with DEAL
# We first illustrate Gaussian denoising with DEAL. The model is applied with a
# denoising operator, and the plotted mask corresponds to the spatially varying
# regularization weights from the last iteration.

physics_denoise = Denoising(GaussianNoise(sigma=sigma)).to(device)
y_denoise = physics_denoise(x)

with torch.no_grad():
    x_hat_denoise = model(y_denoise, physics_denoise)
    mask_denoise = model.model.mask.mean(dim=1, keepdim=True)


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
# Reconstruction example with inpainting
# We next illustrate a simple inpainting problem where a random subset of
# pixels is removed. DEAL combines data fidelity and its learned adaptive
# regularization to recover the missing content.

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
    mask_inpaint = model.mask.mean(dim=1, keepdim=True)

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
