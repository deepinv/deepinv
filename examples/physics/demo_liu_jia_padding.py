import math
import deepinv as dinv

import torch
import torch.fft


def _dst1(x: torch.Tensor, *, dim: int = -1, inverse: bool) -> torch.Tensor:
    r"""
    Compute the one-dimensional discrete sine transform of type I (DST-I) or its inverse (IDST-I)

    This implementation computes the real discrete Fourier transform of the input tensor along one of its dimensions using the formula:

    .. math::

        \mathrm{DST-I}(x_k) = - \frac{1}{2} \Im(\mathrm{DFT}(y_(k+1))),

    .. math::

        \mathrm{IDST-I}(x_k) = - \frac{1}{N + 1} \Im(\mathrm{DFT}(y_(k+1))).

    :param torch.Tensor x: Input tensor.
    :param int dim: Dimension along which to compute the transform. Default is -1 (the last dimension).
    :param bool inverse: If True, compute the inverse DST-I (IDST-I). If False, compute the DST-I.
    :return: Transformed tensor.
    """
    # Compute the DST-I or IDST-I from the DFT of y
    N = x.shape[dim]

    # Compute y the odd extension of x
    # y = (0, x_1, ..., x_N, 0, -x_N, ..., -x_1)
    shape_zeros = list(x.shape)
    shape_zeros[dim] = 1
    zeros = torch.zeros(shape_zeros, dtype=x.dtype, device=x.device)
    x_flipped = torch.flip(x, dims=[dim])
    y = torch.cat([zeros, x, zeros, -x_flipped], dim=dim)

    # Compute the DFT of y
    y = torch.fft.rfft(y, dim=dim, norm="backward")
    y = y.narrow(dim, 1, N)

    # Set the coefficient for forward and inverse transforms
    if not inverse:
        c = -1 / 2
    else:
        c = -1 / (N + 1)

    # Compute the transform
    return c * y.imag


def dst1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""
    Compute the one-dimensional discrete sine transform of type I (DST-I)

    This implementation computes the real discrete Fourier transform of the input tensor along one of its dimensions using the formula:

    .. math::

        \mathrm{DST-I}(x_k) = - \frac{1}{2} \Im(\mathrm{DFT}(y_(k+1))).

    :param torch.Tensor x: Input tensor.
    :param int dim: Dimension along which to compute the transform. Default is -1 (the last dimension).
    :return: Transformed tensor.
    """
    return _dst1(x, dim=dim, inverse=False)


def idst1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""
    Compute the one-dimensional inverse discrete sine transform of type I (IDST-I)

    This implementation computes the real discrete Fourier transform of the input tensor along one of its dimensions using the formula:

    .. math::

        \mathrm{IDST-I}(x_k) = - \frac{1}{N + 1} \Im(\mathrm{DFT}(y_(k+1))).

    :param torch.Tensor x: Input tensor.
    :param int dim: Dimension along which to compute the transform. Default is -1 (the last dimension).
    :return: Transformed tensor.
    """
    return _dst1(x, dim=dim, inverse=True)


def liu_jia_pad(
    x: torch.Tensor, *, padding: tuple[int, int], alpha: int = 1
) -> torch.Tensor:
    """
    Liu-Jia Padding

    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008

    :param torch.Tensor x: Input tensor of shape (B, C, H, W)
    :param tuple(int, int) padding: Tuple specifying the amount of padding (pad_h, pad_w)
    :param int alpha: Border width for Liu-Jia padding (default: 1)
    :return: Padded tensor of shape (B, C, H + 2 * pad_h, W + 2 * pad_w)
    """
    if x.ndim != 4:
        raise ValueError("Input tensor must be 4-dimensional (B, C, H, W)")

    padding_h = 2 * padding[0]
    padding_w = 2 * padding[1]

    BC = tuple(x.shape[:-2])
    H, W = x.shape[-2:]

    A = torch.zeros(BC + (2 * alpha + padding_h, W))
    A[..., :alpha, :] = x[..., -alpha:, :]
    A[..., -alpha:, :] = x[..., :alpha, :]
    a = torch.arange(padding_h) / (padding_h - 1)
    a = a.view((1,) * len(BC) + a.shape)
    A[..., alpha:-alpha, 0] = (1 - a) * A[..., alpha - 1, 0, None] + a * A[
        ..., -alpha, 0, None
    ]
    A[..., alpha:-alpha, -1] = (1 - a) * A[..., alpha - 1, -1, None] + a * A[
        ..., -alpha, -1, None
    ]

    B = torch.zeros(BC + (H, 2 * alpha + padding_w))
    B[..., :, :alpha] = x[..., :, -alpha:]
    B[..., :, -alpha:] = x[..., :, :alpha]
    b = torch.arange(padding_w) / (padding_w - 1)
    b = b.view((1,) * len(BC) + b.shape)
    B[..., 0, alpha:-alpha] = (1 - b) * B[..., 0, alpha - 1, None] + b * B[
        ..., 0, -alpha, None
    ]
    B[..., -1, alpha:-alpha] = (1 - b) * B[..., -1, alpha - 1, None] + b * B[
        ..., -1, -alpha, None
    ]

    if alpha == 1:
        A = compute_padding(A)
        B = compute_padding(B)
    else:
        A[..., alpha - 1 : -alpha + 1, :] = compute_padding(
            A[..., alpha - 1 : -alpha + 1, :]
        )
        B[..., :, alpha - 1 : -alpha + 1] = compute_padding(
            B[..., :, alpha - 1 : -alpha + 1]
        )

    C = torch.zeros(BC + (2 * alpha + padding_h, 2 * alpha + padding_w))
    C[..., :alpha, :] = B[..., -alpha:, :]
    C[..., -alpha:, :] = B[..., :alpha, :]
    C[..., :, :alpha] = A[..., :, -alpha:]
    C[..., :, -alpha:] = A[..., :, :alpha]

    if alpha == 1:
        C = compute_padding(C)
    else:
        C[..., alpha - 1 : -alpha + 1, alpha - 1 : -alpha + 1] = compute_padding(
            C[..., alpha - 1 : -alpha + 1, alpha - 1 : -alpha + 1]
        )

    # Combine the original image and the padding images to form the final padded image
    A = A[..., alpha - 1 : -alpha - 1, :]
    B = B[..., :, alpha:-alpha]
    C = C[..., alpha:-alpha, alpha:-alpha]
    zB = torch.cat((x, B), dim=-1)
    AC = torch.cat((A, C), dim=-1)
    zBAC = torch.cat((zB, AC), dim=-2)

    # Center the original image
    zBAC = zBAC.roll(shifts=padding, dims=(-2, -1))

    return zBAC


def compute_padding(x: torch.Tensor) -> torch.Tensor:
    H, W = x.shape[-2:]

    # Set the inner points to zero
    x[..., 1:-1, 1:-1] = 0

    # Laplacian
    # boundary image contains image intensities at boundaries
    laplacian = torch.zeros_like(x)
    laplacian_bp = torch.zeros_like(x)
    laplacian_bp[..., 1 : H - 1, 1 : W - 1] = (
        x[..., 1:-1, 2:]
        + x[..., 1:-1, :-2]
        + x[..., 2:, 1:-1]
        + x[..., :-2, 1:-1]
        - 4 * x[..., 1:-1, 1:-1]
    )

    laplacian = laplacian - laplacian_bp  # subtract boundary points contribution

    # DST Sine Transform algo starts here
    laplacian = laplacian[..., 1:-1, 1:-1]

    # compute sine tranform
    laplacian = dst1(laplacian, dim=-2)
    laplacian = dst1(laplacian, dim=-1)

    # compute Eigen Values
    u = torch.arange(1, H - 1)
    v = torch.arange(1, W - 1)
    u, v = torch.meshgrid(u, v, indexing="ij")
    laplacian = laplacian / (
        (2 * torch.cos(torch.pi * u / (H - 1)) - 2)
        + (2 * torch.cos(torch.pi * v / (W - 1)) - 2)
    )

    # compute Inverse Sine Transform
    laplacian = idst1(laplacian, dim=-2)
    laplacian = idst1(laplacian, dim=-1)

    # put solution in inner points; outer points obtained from boundary image
    x[..., 1:-1, 1:-1] = laplacian
    return x


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
        y = liu_jia_pad(y, padding=padding)
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
