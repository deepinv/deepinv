r"""
HDR imaging: tone mapping, clipping and quantization
====================================================

Cameras store high dynamic range (HDR) images as low dynamic range (LDR) images by applying a tone mapping curve,
clipping and quantization. We model this with :class:`deepinv.physics.ToneMapping`, :class:`deepinv.physics.Saturation`
and :class:`deepinv.physics.Quantization`, and recover the HDR image in two problems: inverse tone mapping and
deblurring with clipped highlights.

We report the linear PSNR, dominated by the brightest pixels, and :class:`deepinv.loss.metric.MuLawPSNR`, the standard
HDR metric, which weighs dark and bright regions more evenly.
"""

import math

import torch

import deepinv as dinv
from deepinv.physics.functional import srgb_eotf, srgb_oetf

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# %%
# Load the HDR image
# ------------------
# Linear HDR image, scaled so that its highlights are up to 5 times the sRGB clipping level.

x_max = 10.0
x = dinv.utils.load_example("forest_path_hdr.pt", device=device) * x_max

denoiser = dinv.models.DRUNet(pretrained="download", device=device)
psnr = dinv.metric.PSNR(max_pixel=x_max)
mu_psnr = dinv.metric.MuLawPSNR(max_pixel=x_max)

# %%
# Inverse tone mapping
# --------------------
# A logarithmic curve maps the HDR range to :math:`[0, 1]` without clipping, followed by 6-bit quantization.
# Inverting the curve amplifies the quantization error in the highlights.
# As quantization noise is roughly uniform in the tone mapped domain, we denoise before inverting the curve.

levels = 64
physics = dinv.physics.ToneMapping(
    curve="log",
    curve_params={"L": x_max},
    sensor_model=dinv.physics.Quantization(levels=levels),
)
y = physics(x)

sigma = 1 / (math.sqrt(12) * (levels - 1))  # std of uniform quantization noise
with torch.no_grad():
    x_inv = physics.A_dagger(y)
    x_den = physics.A_dagger(denoiser(y, sigma))

dinv.utils.plot(
    [srgb_oetf(x), srgb_oetf(x_inv), srgb_oetf(x_den)],
    titles=["Ground truth", "Inverse curve", "Denoised + inverse curve"],
    subtitles=[""]
    + [
        f"PSNR {psnr(x_hat, x).item():.2f}, $\\mu$-law {mu_psnr(x_hat, x).item():.2f} dB"
        for x_hat in (x_inv, x_den)
    ],
    rescale_mode="clip",
    figsize=(12, 4),
)

# %%
# Deblurring with clipped highlights
# ----------------------------------
# Blur and Poisson noise act on the linear image, then the sensor applies the sRGB curve, clipping and 8-bit
# quantization: :math:`y = \mathrm{round}(\mathrm{clip}(\mathrm{OETF}(\mathcal{P}(Ax))))`.

psf = dinv.physics.functional.gaussian_blur(psf_size=(15, 15), sigma=1.5, device=device)
quantizer = dinv.physics.Quantization(levels=256)
camera = dinv.physics.BlurFFT(
    img_size=x.shape[1:],
    filter=psf,
    device=device,
    noise_model=dinv.physics.PoissonNoise(gain=1e-3),
    sensor_model=lambda z: quantizer(srgb_oetf(z)),
)
y = camera(x)

# %%
# We use plug-and-play proximal gradient descent with DRUNet, which is trained on sRGB images. We therefore
# reconstruct the sRGB image :math:`u = \mathrm{OETF}(x)` through the composed physics
# :math:`u \mapsto \mathrm{OETF}(A\,\mathrm{EOTF}(u))`.
# Data-fidelity terms ignore the sensor model, so we add clipping to the last curve: clipped pixels then only
# penalize estimates below the clipping level. We compare with a reconstruction that ignores clipping.

blur = dinv.physics.BlurFFT(img_size=x.shape[1:], filter=psf, device=device)
decode = dinv.physics.ToneMapping(curve=srgb_eotf, inverse=srgb_oetf)
saturation = dinv.physics.Saturation()
physics_ignored = dinv.physics.compose(decode, blur, dinv.physics.ToneMapping("srgb"))
physics_aware = dinv.physics.compose(
    decode, blur, dinv.physics.ToneMapping(curve=lambda z: saturation(srgb_oetf(z)))
)

n_iter = 40
model = dinv.optim.PGD(
    prior=dinv.optim.PnP(denoiser),
    data_fidelity=dinv.optim.L2(),
    params_algo={
        "stepsize": 1.0,
        "g_param": torch.logspace(math.log10(0.1), math.log10(0.02), n_iter).tolist(),
        "lambda": 1.0,
    },
    max_iter=n_iter,
)

with torch.no_grad():
    x_ignored = srgb_eotf(model(y, physics_ignored, init=y))
    x_aware = srgb_eotf(model(y, physics_aware, init=y))

# %%
# We display the images at normal exposure, then 3 stops darker to show the clipped highlights.

x_meas = srgb_eotf(y)
for exposure in (1, 1 / 8):
    dinv.utils.plot(
        [srgb_oetf(img * exposure) for img in (x, x_meas, x_ignored, x_aware)],
        titles=["Ground truth", "Measurement", "Clipping ignored", "Clipping-aware"],
        subtitles=[""]
        + [
            f"PSNR {psnr(x_hat, x).item():.2f}, $\\mu$-law {mu_psnr(x_hat, x).item():.2f} dB"
            for x_hat in (x_meas, x_ignored, x_aware)
        ],
        rescale_mode="clip",
        figsize=(16, 4),
    )
