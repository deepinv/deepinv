r"""
Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding
=================================================================

Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically
blurred using circular filters. This makes the use of spectral deconvolution methods (inverse
filtering, Wiener filtering) impractical and prone to ringing artifacts.
:func:`Liu-Jia padding <deepinv.physics.functional.liu_jia_pad>` :footcite:p:`liu2008reducing`
is a pre-processing step that pads the input image to make it have smooth circular boundaries,
while preserving the original spectral content as much as possible.

The implementation used here is adapted from `the one <https://github.com/cszn/USRNet>`_
featured in the work of :footcite:t:`zhang2020deep`.

This demo compares deconvolution with and without Liu-Jia padding, for both inverse filtering
and Wiener filtering, on a realistic blurred image obtained using valid (cropped) convolution instead of circular convolution.
"""

import torch
import deepinv as dinv
import math

# For reproducilibity
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# Select the device
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Load in the test image
x = dinv.utils.load_example("butterfly.png", img_size=256).to(device)

# Define the blur parameters
gaussian_std = 1.0
ksize = 6 * math.ceil(gaussian_std) + 1
kernel = dinv.physics.functional.gaussian_blur(
    psf_size=(ksize, ksize), sigma=gaussian_std, device=device
)
physics = dinv.physics.Blur(filter=kernel, padding="valid", device=device)

# Compute the blurry image
y = physics(x)

# Display both images and the kernel
dinv.utils.plot(
    [x, kernel, y],
    ["Input", "Blur kernel", "Blurry"],
)


# %%
# Spectral deconvolution algorithms
# ---------------------------------
# We evaluate the impact of Liu-Jia padding on two spectral deconvolution methods:
# inverse filtering and Wiener filtering. Both methods assume a circular blur with
#
# .. math::
#
#      Y(f) = H(f) X(f)
#
# where :math:`X` is the Fourier transform of the original image, :math:`Y` is the
# Fourier transform of the blurry image, and :math:`H` is the Fourier transform of
# the blur kernel.
#
# Inverse filtering is implemented by ``BlurFFT.A_dagger`` and
# it is defined by the equation
#
# .. math::
#
#     \hat{X}(f) = \frac{Y(f)}{H(f)}
#
# and Wiener filtering is implemented by ``BlurFFT.prox_l2`` and
# it is defined by the equation
#
# .. math::
#
#    \hat{X}(f) = \frac{H^*(f)}{|H(f)|^2 + 1 / \mathrm{SNR}(f)} Y(f)
#
# where the SNR depends on the power spectral density of the input image and
# noise. In our case we assume it is an arbitrary constant that we tune manually.


class SpectralDeconvolution(dinv.models.Reconstructor):
    def __init__(self, liu_jia_padding: bool, kind: str, gamma: float = 1e4):
        super().__init__()
        self.liu_jia_padding = liu_jia_padding
        self.kind = kind
        self.gamma = gamma

    def forward(self, y: torch.Tensor, physics: dinv.physics.Physics) -> torch.Tensor:
        # Optional padding
        if self.liu_jia_padding:
            H, W = y.shape[-2:]
            padding = (H // 4, W // 4)
            y = dinv.physics.functional.liu_jia_pad(y, padding=padding)
            margin = (
                (y.shape[-2] - H) // 2,
                (y.shape[-1] - W) // 2,
            )
        else:
            margin = None

        # Spectral deconvolution
        kernel = physics.filter
        physics_circular = dinv.physics.BlurFFT(
            filter=kernel, img_size=y.shape[1:], device=y.device
        )
        match self.kind:
            case "inverse":
                x_hat = physics_circular.A_dagger(y)
            case "wiener":
                x_hat = physics_circular.prox_l2(y=y, z=0, gamma=self.gamma)
            case _:
                raise ValueError(f"Unknown deconvolution kind: {self.kind}")

        # Clipping to attenuate artifacts
        x_hat = x_hat.clip(0, 1)

        # Cropping if Liu-Jia padding was used
        if margin is not None:
            x_hat = x_hat[..., margin[0] : -margin[0], margin[1] : -margin[1]]

        return x_hat


model_inv_pad = SpectralDeconvolution(liu_jia_padding=True, kind="inverse")
model_inv_nopad = SpectralDeconvolution(liu_jia_padding=False, kind="inverse")
model_wiener_pad = SpectralDeconvolution(liu_jia_padding=True, kind="wiener")
model_wiener_nopad = SpectralDeconvolution(liu_jia_padding=False, kind="wiener")

# %%
# Results
# -------
# Liu-Jia padding gives better performance (PSNR) and produces images with less
# severe ringing artifacts compared to the baselines.

# Define the metric
psnr_fn = dinv.metric.PSNR(center_crop=y.shape[-2:])

# Estimate the sharp image for each model
x_hat_inv_pad = model_inv_pad(y, physics)
x_hat_inv_nopad = model_inv_nopad(y, physics)
x_hat_wiener_pad = model_wiener_pad(y, physics)
x_hat_wiener_nopad = model_wiener_nopad(y, physics)

# Compute the performance metrics
psnr_blurry = psnr_fn(y, x).item()
psnr_inv_pad = psnr_fn(x_hat_inv_pad, x).item()
psnr_inv_nopad = psnr_fn(x_hat_inv_nopad, x).item()
psnr_wiener_pad = psnr_fn(x_hat_wiener_pad, x).item()
psnr_wiener_nopad = psnr_fn(x_hat_wiener_nopad, x).item()

# Plot results for inverse filtering
dinv.utils.plot(
    [x, y, x_hat_inv_pad, x_hat_inv_nopad],
    ["GT", "Blurry", "Liu-Jia", "Regular"],
    subtitles=[
        "",
        f"{psnr_blurry:.1f} dB",
        f"{psnr_inv_pad:.1f} dB",
        f"{psnr_inv_nopad:.1f} dB",
    ],
    suptitle="Deblurring with Inverse Filtering",
)

# Plot results for Wiener filtering
dinv.utils.plot(
    [x, y, x_hat_wiener_pad, x_hat_wiener_nopad],
    ["GT", "Blurry", "Liu-Jia", "Regular"],
    subtitles=[
        "",
        f"{psnr_blurry:.1f} dB",
        f"{psnr_wiener_pad:.1f} dB",
        f"{psnr_wiener_nopad:.1f} dB",
    ],
    suptitle="Deblurring with Wiener Filtering",
)

# %%
# :References:
#
# .. footbibliography::
