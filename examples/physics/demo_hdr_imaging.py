r"""
HDR imaging: tone mapping, clipping and quantization
====================================================

Cameras and displays store high dynamic range (HDR) linear images :math:`x` as
low dynamic range (LDR) images, by applying a tone mapping curve, clipping and
quantization. This demo models this acquisition chain with
:class:`deepinv.physics.ToneMapping` and the sensor models
:class:`deepinv.physics.Saturation` and :class:`deepinv.physics.Quantization`,
and recovers the HDR image in two problems:

1. inverse tone mapping of a coarsely quantized image,
2. deblurring of a noisy image with clipped highlights.

Results are evaluated with the PSNR in the linear domain, which is dominated by
the brightest pixels, and with :class:`deepinv.loss.metric.MuLawPSNR`, which
weighs dark and bright regions more evenly.
"""

import math

import torch

import deepinv as dinv
from deepinv.physics.functional import srgb_eotf, srgb_oetf

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# %%
# Load the HDR image
# -------------------------------------------------------
# A linear HDR image, scaled so that its brightest highlights are about 5 times
# above the clipping level of an sRGB sensor. The scale ``x_max`` is the peak
# value of the full image, from which this crop is taken.

x_max = 10.0
x = dinv.utils.load_example("forest_path_hdr.pt", device=device) * x_max

denoiser = dinv.models.DRUNet(pretrained="download", device=device)
psnr = dinv.metric.PSNR(max_pixel=x_max)
mu_psnr = dinv.metric.MuLawPSNR(max_pixel=x_max)


def evaluate(estimates):
    for name, x_est in estimates.items():
        print(
            f"{name:28s} PSNR {psnr(x_est, x).item():.2f} dB, "
            f"mu-law PSNR {mu_psnr(x_est, x).item():.2f} dB"
        )


def show(images, exposure=1.0):
    dinv.utils.plot(
        {name: srgb_oetf(img * exposure) for name, img in images.items()},
        rescale_mode="clip",
        figsize=(4 * len(images), 4),
    )


# %%
# Inverse tone mapping
# -------------------------------------------------------
# The logarithmic curve compresses the whole HDR range into :math:`[0, 1]`
# without clipping, and the result is quantized to 6 bits:
# :math:`y = \mathrm{round}(\mathrm{TM}(x))`. Inverting the curve amplifies the
# quantization error in the highlights, where the inverse curve is steep.
# Quantization noise is approximately uniform in the tone mapped domain, so we
# remove it with a denoiser before inverting the curve.

levels = 64
physics = dinv.physics.ToneMapping(
    curve="log",
    curve_params={"L": x_max},
    sensor_model=dinv.physics.Quantization(levels=levels),
)
y = physics(x)

sigma = 1 / (math.sqrt(12) * (levels - 1))  # std of the quantization noise
with torch.no_grad():
    x_itm = {
        "Inverse curve": physics.A_dagger(y),
        "Denoised, then inverse curve": physics.A_dagger(denoiser(y, sigma)),
    }
evaluate(x_itm)
show({"Ground truth": x, **x_itm})

# %%
# Deblurring with clipped highlights
# -------------------------------------------------------
# The blur and the Poisson noise act on the linear signal, then the sensor
# applies the sRGB curve, clipping and 8-bit quantization:
# :math:`y = \mathrm{round}(\mathrm{clip}(\mathrm{OETF}(\mathcal{P}(Ax))))`.

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
# We reconstruct with plug-and-play proximal gradient descent. DRUNet is trained
# on sRGB images, so we reconstruct the sRGB image :math:`u = \mathrm{OETF}(x)`:
# the physics seen by the reconstruction decodes :math:`u` with a custom-curve
# tone mapping, blurs it and encodes it again. Data-fidelity terms only use the
# forward operator, so the clipping is included in the last curve: clipped
# pixels then only penalize estimates below the clipping level. For comparison,
# we also reconstruct while ignoring the clipping.

blur = dinv.physics.BlurFFT(img_size=x.shape[1:], filter=psf, device=device)
decode = dinv.physics.ToneMapping(curve=srgb_eotf, inverse=srgb_oetf)
saturation = dinv.physics.Saturation()
encoders = {
    "Clipping ignored": dinv.physics.ToneMapping(curve="srgb"),
    "Clipping-aware": dinv.physics.ToneMapping(
        curve=lambda z: saturation(srgb_oetf(z))
    ),
}

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

x_deblur = {"Measurement": srgb_eotf(y)}
for name, encode in encoders.items():
    with torch.no_grad():
        u = model(y, dinv.physics.compose(decode, blur, encode), init=y)
    x_deblur[name] = srgb_eotf(u)
evaluate(x_deblur)

# %%
# We display the images at normal exposure, then 3 stops darker to reveal the
# highlights that are clipped in the measurement.

show({"Ground truth": x, **x_deblur})
show({"Ground truth": x, **x_deblur}, exposure=1 / 8)
