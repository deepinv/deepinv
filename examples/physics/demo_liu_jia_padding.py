import deepinv as dinv

import torch

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


def liu_jia_pad(z, padding, *, marginp1: int = 1):
    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    padding_h = 2 * padding[0]
    padding_w = 2 * padding[1]

    BC = tuple(z.shape[:-2])
    H, W = z.shape[-2:]

    A = torch.zeros(BC + (2 * marginp1 + padding_h, W))
    A[..., :marginp1, :] = z[..., -marginp1:, :]
    A[..., -marginp1:, :] = z[..., :marginp1, :]
    a = torch.arange(padding_h) / (padding_h - 1)
    a = a.view((1,) * len(BC) + a.shape)
    A[..., marginp1:-marginp1, 0] = (1 - a) * A[..., marginp1 - 1, 0, None] + a * A[
        ..., -marginp1, 0, None
    ]
    A[..., marginp1:-marginp1, -1] = (1 - a) * A[..., marginp1 - 1, -1, None] + a * A[
        ..., -marginp1, -1, None
    ]

    B = torch.zeros(BC + (H, 2 * marginp1 + padding_w))
    B[..., :, :marginp1] = z[..., :, -marginp1:]
    B[..., :, -marginp1:] = z[..., :, :marginp1]
    b = torch.arange(padding_w) / (padding_w - 1)
    b = b.view((1,) * len(BC) + b.shape)
    B[..., 0, marginp1:-marginp1] = (1 - b) * B[..., 0, marginp1 - 1, None] + b * B[
        ..., 0, -marginp1, None
    ]
    B[..., -1, marginp1:-marginp1] = (1 - b) * B[..., -1, marginp1 - 1, None] + b * B[
        ..., -1, -marginp1, None
    ]

    if marginp1 == 1:
        A = solve_min_laplacian(A)
        B = solve_min_laplacian(B)
    else:
        A[..., marginp1 - 1 : -marginp1 + 1, :] = solve_min_laplacian(
            A[..., marginp1 - 1 : -marginp1 + 1, :]
        )
        B[..., :, marginp1 - 1 : -marginp1 + 1] = solve_min_laplacian(
            B[..., :, marginp1 - 1 : -marginp1 + 1]
        )

    C = torch.zeros(BC + (2 * marginp1 + padding_h, 2 * marginp1 + padding_w))
    C[..., :marginp1, :] = B[..., -marginp1:, :]
    C[..., -marginp1:, :] = B[..., :marginp1, :]
    C[..., :, :marginp1] = A[..., :, -marginp1:]
    C[..., :, -marginp1:] = A[..., :, :marginp1]

    if marginp1 == 1:
        C = solve_min_laplacian(C)
    else:
        C[..., marginp1 - 1 : -marginp1 + 1, marginp1 - 1 : -marginp1 + 1] = (
            solve_min_laplacian(
                C[..., marginp1 - 1 : -marginp1 + 1, marginp1 - 1 : -marginp1 + 1]
            )
        )

    A = A[..., marginp1 - 1 : -marginp1 - 1, :]
    B = B[..., :, marginp1:-marginp1]
    C = C[..., marginp1:-marginp1, marginp1:-marginp1]
    zB = torch.cat((z, B), dim=-1)
    AC = torch.cat((A, C), dim=-1)
    zBAC = torch.cat((zB, AC), dim=-2)
    zBAC = zBAC.roll(shifts=padding, dims=(-2, -1))
    return zBAC


def solve_min_laplacian(mat):
    H, W = mat.shape[-2:]

    mat[..., 1:-1, 1:-1] = 0

    # Laplacian
    # boundary image contains image intensities at boundaries
    laplacian = torch.zeros_like(mat)
    laplacian_bp = torch.zeros_like(mat)
    laplacian_bp[..., 1 : H - 1, 1 : W - 1] = (
        mat[..., 1:-1, 2:]
        + mat[..., 1:-1, :-2]
        + mat[..., 2:, 1:-1]
        + mat[..., :-2, 1:-1]
        - 4 * mat[..., 1:-1, 1:-1]
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
    mat[..., 1:-1, 1:-1] = laplacian
    return mat


device = "cpu"
x = dinv.utils.load_example("butterfly.png", img_size=64).to(device)

# Define blur kernel and physics
kernel = torch.tensor(
    [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]],
    device=device,
)
kernel /= kernel.sum()
kernel = kernel.unsqueeze(0).unsqueeze(0)
physics = dinv.physics.Blur(filter=kernel, padding="valid")
y = physics(x)

# Crop for comparison
if kernel.shape[-2] % 2 != 1 or kernel.shape[-1] % 2 != 1:
    raise ValueError("Kernel size is expected to be odd")

margin = (
    (kernel.shape[-2] - 1) // 2,
    (kernel.shape[-1] - 1) // 2,
)
x = x[..., margin[0] : -margin[0], margin[1] : -margin[1]]

# Liu-Jia Padding
H, W = y.shape[-2:]
padding = (H // 4, W // 4)
y = liu_jia_pad(y, padding=padding)

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
x_hat = x_hat / (otf + 1e-3)
# 5. Compute the inverse DFT
x_hat = torch.fft.ifft2(x_hat).real
# 6. Clip and quantize
x_hat = torch.clamp(x_hat, 0, 1)
x_hat = torch.round(x_hat * 255) / 255

# Cropping
margin = (
    (y.shape[-2] - H) // 2,
    (y.shape[-1] - W) // 2,
)
y = y[..., margin[0] : -margin[0], margin[1] : -margin[1]]
x_hat = x_hat[..., margin[0] : -margin[0], margin[1] : -margin[1]]

if x.shape != y.shape:
    raise ValueError("Shapes do not match after cropping")

psnr_fn = dinv.metric.PSNR()
psnr = psnr_fn(y, x).item()
psnr_x_hat = psnr_fn(x_hat, x).item()

dinv.utils.plot(
    [x, y, x_hat], ["x", f"y ({psnr:.1f} dB)", f"x_hat ({psnr_x_hat:.1f} dB)"]
)
