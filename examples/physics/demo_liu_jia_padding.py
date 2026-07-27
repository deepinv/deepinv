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

device = "cpu"
x = dinv.utils.load_example("butterfly.png", img_size=256).to(device)

gaussian_std = 1.0
ksize = 6 * math.ceil(gaussian_std) + 1
kernel = dinv.physics.functional.gaussian_blur(
    psf_size=(ksize, ksize), sigma=gaussian_std, device=device
)
physics = dinv.physics.Blur(filter=kernel, padding="valid")

y = physics(x)

# Crop the ground truth to match the valid-convolution output of the blur
margin = (
    (kernel.shape[-2] - 1) // 2,
    (kernel.shape[-1] - 1) // 2,
)
x = x[..., margin[0] : -margin[0], margin[1] : -margin[1]]

dinv.utils.plot(
    [x, kernel, y],
    ["Input", "Blur kernel", "Blurry"],
)

# %%
# Deconvolution
# -------------


def deblur(
    y: torch.Tensor,
    *,
    kernel: torch.Tensor,
    liu_jia_padding: bool,
    deconvolution_kind: str,
    eps: float = 1e-3,
) -> torch.Tensor:
    # Liu-Jia Padding
    if liu_jia_padding:
        H, W = y.shape[-2:]
        padding = (H // 4, W // 4)
        y = dinv.physics.functional.liu_jia_pad(y, padding=padding)
        margin = (
            (y.shape[-2] - H) // 2,
            (y.shape[-1] - W) // 2,
        )
    else:
        margin = None

    # Deconvolution
    # 1. Pad k to make it the size of y with the central tap at (0,0)
    k = torch.nn.functional.pad(
        kernel,
        (
            0,
            y.shape[-1] - kernel.shape[-1],
            0,
            y.shape[-2] - kernel.shape[-2],
        ),
    )
    k = k.roll(shifts=(-(kernel.shape[-2] // 2), -(kernel.shape[-1] // 2)), dims=(2, 3))
    # 2. Compute the OTF
    otf = torch.fft.fft2(k)
    # 3. Compute the DFT of y
    x_hat = torch.fft.fft2(y)
    # 4. Apply the inverse filter formula
    if deconvolution_kind == "inverse":
        x_hat = x_hat / (otf + eps)
    elif deconvolution_kind == "wiener":
        x_hat = torch.conj(otf) * x_hat / (torch.abs(otf) ** 2 + eps)
    elif deconvolution_kind == "richardson-lucy":
        raise NotImplementedError("Richardson-Lucy deconvolution is not implemented")
    else:
        raise ValueError(f"Unknown filter kind: {deconvolution_kind}")
    # 5. Compute the inverse DFT
    x_hat = torch.fft.ifft2(x_hat).real
    # 6. Clip
    x_hat = torch.clamp(x_hat, 0, 1)
    # 7. Quantize
    x_hat = torch.round(x_hat * 255) / 255

    # Cropping
    if margin is not None:
        x_hat = x_hat[..., margin[0] : -margin[0], margin[1] : -margin[1]]

    return x_hat


# Comparisons
psnr_fn = dinv.metric.PSNR()
base_psnr = psnr_fn(y, x).item()

# Compare Liu-Jia padding vs no padding for inverse filtering
x_hat_liu_jia_padding = deblur(
    y, kernel=kernel, liu_jia_padding=True, deconvolution_kind="inverse", eps=1e-1
)
x_hat_no_padding = deblur(
    y, kernel=kernel, liu_jia_padding=False, deconvolution_kind="inverse", eps=1e-1
)

psnr_liu_jia_padding = psnr_fn(x_hat_liu_jia_padding, x).item()
psnr_no_padding = psnr_fn(x_hat_no_padding, x).item()

dinv.utils.plot(
    [x, y, x_hat_liu_jia_padding, x_hat_no_padding],
    [
        f"GT",
        f"Blurry {base_psnr:.1f} dB",
        f"Liu-Jia Padding {psnr_liu_jia_padding:.1f} dB",
        f"No Padding {psnr_no_padding:.1f} dB",
    ],
)

# Compare Liu-Jia padding vs no padding for Wiener deconvolution
x_hat_liu_jia_padding = deblur(
    y, kernel=kernel, liu_jia_padding=True, deconvolution_kind="wiener"
)
x_hat_no_padding = deblur(
    y, kernel=kernel, liu_jia_padding=False, deconvolution_kind="wiener"
)

psnr_liu_jia_padding = psnr_fn(x_hat_liu_jia_padding, x).item()
psnr_no_padding = psnr_fn(x_hat_no_padding, x).item()

dinv.utils.plot(
    [x, y, x_hat_liu_jia_padding, x_hat_no_padding],
    [
        f"GT",
        f"Blurry {base_psnr:.1f} dB",
        f"Liu-Jia Padding {psnr_liu_jia_padding:.1f} dB",
        f"No Padding {psnr_no_padding:.1f} dB",
    ],
)
