"""
Real-World Non-blind Image Deblurring with Liu-Jia Padding
"""

import torch
import deepinv as dinv
import math

device = "cpu"
x = dinv.utils.load_example("butterfly.png", img_size=256).to(device)

# Define blur kernel and physics
gaussian_std = 1.0
ksize = 6 * math.ceil(gaussian_std) + 1
u = torch.arange(ksize, device=device) - ksize // 2
v = torch.arange(ksize, device=device) - ksize // 2
u, v = torch.meshgrid(u, v, indexing="ij")
kernel = torch.exp(-(u**2 + v**2) / (2 * gaussian_std**2))
kernel /= kernel.sum()
kernel = kernel.unsqueeze(0).unsqueeze(0)
physics = dinv.physics.Blur(filter=kernel, padding="valid")


# Compute the blur in linear sRGB space
def eotf(x: torch.Tensor) -> torch.Tensor:
    # Map from sRGB to linear sRGB
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )


def oetf(x: torch.Tensor) -> torch.Tensor:
    # Map from linear sRGB to sRGB
    return torch.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * (x ** (1 / 2.4)) - 0.055,
    )


y = physics(eotf(x))

# Crop for comparison
if kernel.shape[-2] % 2 != 1 or kernel.shape[-1] % 2 != 1:
    raise ValueError("Kernel size is expected to be odd")

margin = (
    (kernel.shape[-2] - 1) // 2,
    (kernel.shape[-1] - 1) // 2,
)
x = x[..., margin[0] : -margin[0], margin[1] : -margin[1]]


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
        y = dinv.utils.liu_jia_pad(y, padding=padding)
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
    # 7. Move to sRGB space
    x_hat = oetf(x_hat)
    # 8. Quantize
    x_hat = torch.round(x_hat * 255) / 255

    # Cropping
    if margin is not None:
        x_hat = x_hat[..., margin[0] : -margin[0], margin[1] : -margin[1]]

    return x_hat


# Comparisons
psnr_fn = dinv.metric.PSNR()
base_psnr = psnr_fn(oetf(y), x).item()

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
    [x, oetf(y), x_hat_liu_jia_padding, x_hat_no_padding],
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
    [x, oetf(y), x_hat_liu_jia_padding, x_hat_no_padding],
    [
        f"GT",
        f"Blurry {base_psnr:.1f} dB",
        f"Liu-Jia Padding {psnr_liu_jia_padding:.1f} dB",
        f"No Padding {psnr_no_padding:.1f} dB",
    ],
)
