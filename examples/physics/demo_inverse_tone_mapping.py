r"""
Inverse tone mapping via dequantization
======================================

This demo shows the :class:`deepinv.physics.ToneMapping` operator with a
:class:`deepinv.physics.Quantization` sensor, modelling a low dynamic range
(LDR) acquisition chain

.. math::

    y = \mathrm{round}\left(\mathrm{TM}(x)\right)

applied to a high dynamic range (HDR) image, where :math:`\mathrm{TM}` is a
bounded tone mapping curve (here, the logarithmic curve). Since the curve is
bounded, no clipping occurs: the whole HDR range is compressed injectively
into :math:`[0, 1]`, and the quantization of the tone mapped image is the only
degradation. Recovering the HDR image :math:`x` from :math:`y` is an inverse
tone mapping problem, whose difficulty comes from the quantization: one least
significant bit of :math:`y` corresponds to a large linear-domain error in
the highlights, where the inverse curve is steep.

A pretrained sRGB denoiser can be used as prior via a change of variables:
quantization noise is approximately uniform in the tone mapped domain, so the
problem becomes a plain denoising problem on the tone mapped image, where
standard pretrained denoisers apply. The reconstruction is then mapped back
to the linear domain with the inverse curve.
"""

# %%
# Imports and setup
# -------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
import torch

import deepinv as dinv
import math
from deepinv.physics.functional import log_tone, log_tone_inverse

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load the HDR image (Radiance RGBE format, linear values up to ~8000)
# -------------------------------------------------------

x = cv2.imread("forest_path.hdr", flags=cv2.IMREAD_UNCHANGED)[..., ::-1]  # BGR->RGB
x = torch.from_numpy(x.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
x = torch.nn.functional.interpolate(
    x, scale_factor=0.25, mode="bilinear", align_corners=False
)
x = x / x.max() * 100.0  # normalize to a comfortable HDR range [0, 100]

# %%
# Set up the physics: logarithmic tone mapping + 6-bit quantization
# -------------------------------------------------------

L = 100.0  # the whole HDR range [0, 100] maps to [0, 1]
levels = 64  # 6-bit ADC

physics = dinv.physics.ToneMapping(
    curve="log",
    curve_params={"L": L},
    sensor_model=dinv.physics.Quantization(levels=levels),
)
y = physics(x)

# %%
# Baseline: naive inversion of the tone mapping curve
# -------------------------------------------------------

x_naive = physics.A_dagger(y)

# %%
# Reconstruction: denoise in the tone mapped domain, then transform back
# -------------------------------------------------------
# Quantization noise is approximately uniform with standard deviation
# sigma = 1 / (sqrt(12) * (levels - 1)), so the tone mapped measurement y is
# just a noisy sRGB-like image: any pretrained denoiser applies.

denoiser = dinv.models.DRUNet(pretrained="download", device=device)

sigma = 1.0 / (12**0.5 * (levels - 1))
u_hat = denoiser(y, sigma)
x_hat = log_tone_inverse(u_hat, L=L)

# %%
# Metrics and plots
# -------------------------------------------------------

psnr = dinv.metric.PSNR(max_pixel=None)
u_gt = log_tone(x, L=L)

print(f"PSNR naive inversion:            {psnr(x_naive, x).item():.2f} dB")
print(f"PSNR denoised reconstruction:    {psnr(x_hat, x).item():.2f} dB")
print(
    f"PSNR tone-mapped domain:         naive {psnr(y, u_gt).item():.2f} dB, denoised {psnr(u_hat, u_gt).item():.2f} dB"
)


def show(img, ax, title, vmax=None):
    img = img[0].permute(1, 2, 0).detach().cpu().clamp(0, vmax).numpy()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")


fig, axes = plt.subplots(1, 4, figsize=(20, 5))
show(log_tone(x, L=L), axes[0], "HDR ground truth (tone mapped)")
show(y, axes[1], f"LDR measurement ({int(math.log2(levels))}-bit)")
show(log_tone(x_naive, L=L), axes[2], "Naive inverse curve")
show(u_hat, axes[3], "Denoised reconstruction (tone mapped)")
plt.tight_layout()
plt.show()
