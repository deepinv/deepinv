import deepinv as dinv

import torch

import torch
import torch.fft


def dst(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the Discrete Sine Transform Type 1 (DST-I).
    Matches scipy.fftpack.dst(type=1).

    The DST-I is equivalent to the imaginary part of the DFT of an
    odd-symmetric extension of the input signal.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the transform. Default is -1.

    Returns:
        torch.Tensor: The DST-I transformed tensor.
    """
    # 1. Setup dimensions
    n = x.shape[dim]

    # 2. Create the odd extension: [0, x, 0, -flip(x)]
    # We need to construct the padding and reversed parts carefully handling dimensions.

    # Slice to get a tensor of shape [..., 1, ...] for zeros
    shape_zeros = list(x.shape)
    shape_zeros[dim] = 1
    zeros = torch.zeros(shape_zeros, dtype=x.dtype, device=x.device)

    # Flip x along the specified dimension
    x_flipped = torch.flip(x, dims=[dim])

    # Construct the odd extension
    # Sequence: [0, x_0, x_1, ..., x_{N-1}, 0, -x_{N-1}, ..., -x_0]
    # Length: 1 + N + 1 + N = 2N + 2
    odd_ext = torch.cat([zeros, x, zeros, -x_flipped], dim=dim)

    # 3. Compute RFFT
    # The RFFT of a real, odd sequence is purely imaginary.
    spec = torch.fft.rfft(odd_ext, dim=dim)

    # 4. Extract DST-I
    # Scipy DST-I definition: y[k] = 2 * sum(x[n] * sin(pi*(k+1)*(n+1)/(N+1)))
    # The Imaginary part of DFT of odd extension gives: -2 * sum(x[n] * sin(...))
    # So we take the negative imaginary part.
    # We slice [1:n+1] because index 0 is DC (0), and we need indices 1..N.

    slices = [slice(None)] * spec.ndim
    slices[dim] = slice(1, n + 1)

    return -spec.imag[tuple(slices)]


def idst(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the Inverse Discrete Sine Transform Type 1 (IDST-I).
    Matches scipy.fftpack.idst(type=1).

    NOTE: scipy.fftpack.idst(type=1) is UNNORMALIZED by default.
    It is identical to dst(type=1).
    To get the true mathematical inverse, the result must be
    scaled by 1 / (2 * (N + 1)).

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the transform. Default is -1.

    Returns:
        torch.Tensor: The IDST-I transformed tensor (unscaled).
    """
    return dst(x, dim=dim)


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
    laplacian = dst(laplacian, dim=-2) / 2
    laplacian = dst(laplacian, dim=-1) / 2

    # compute Eigen Values
    u = torch.arange(1, H - 1)
    v = torch.arange(1, W - 1)
    u, v = torch.meshgrid(u, v, indexing="ij")
    laplacian = laplacian / (
        (2 * torch.cos(torch.pi * u / (H - 1)) - 2)
        + (2 * torch.cos(torch.pi * v / (W - 1)) - 2)
    )

    # compute Inverse Sine Transform
    laplacian = idst(laplacian, dim=-2)
    laplacian = idst(laplacian, dim=-1)
    laplacian = laplacian / (laplacian.shape[-2] + 1)
    laplacian = laplacian / (laplacian.shape[-1] + 1)

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
