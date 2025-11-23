import deepinv as dinv

from scipy import fftpack
import torch



def liu_jia_pad(y, padding):
    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    # (B, C, H, W) -> (BC, H, W)
    BC = y.shape[:2]
    y = y.flatten(start_dim=0, end_dim=1)

    zs = []
    for z in torch.unbind(y, dim=0):
        z = _liu_jia_pad(z, padding)
        zs.append(z)
    y = torch.stack(zs, 0)

    # (BC, H, W) -> (B, C, H, W)
    return y.unflatten(dim=0, sizes=BC)


def _liu_jia_pad(z, padding, *, marginp1: int = 1):
    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if z.ndim != 2:
        raise ValueError("The input tensor must have exactly two dimensions.")

    padding_h = 2 * padding[0]
    padding_w = 2 * padding[1]

    BC = tuple(z.shape[:-2])
    H, W = z.shape[-2:]

    A = torch.zeros(BC + (2 * marginp1 + padding_h, W))
    A[..., :marginp1, :] = z[..., -marginp1:, :]
    A[..., -marginp1:, :] = z[..., :marginp1, :]
    a = torch.arange(padding_h) / (padding_h - 1)
    A[..., marginp1:-marginp1, 0] = (1 - a) * A[..., marginp1 - 1, 0] + a * A[
        ..., -marginp1, 0
    ]
    A[..., marginp1:-marginp1, -1] = (1 - a) * A[..., marginp1 - 1, -1] + a * A[
        ..., -marginp1, -1
    ]

    B = torch.zeros(BC + (H, 2 * marginp1 + padding_w))
    B[..., :, :marginp1] = z[..., :, -marginp1:]
    B[..., :, -marginp1:] = z[..., :, :marginp1]
    a = torch.arange(padding_w) / (padding_w - 1)
    B[..., 0, marginp1:-marginp1] = (1 - a) * B[..., 0, marginp1 - 1] + a * B[
        ..., 0, -marginp1
    ]
    B[..., -1, marginp1:-marginp1] = (1 - a) * B[..., -1, marginp1 - 1] + a * B[
        ..., -1, -marginp1
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
    zB = torch.hstack((z, B))
    AC = torch.hstack((A, C))
    zBAC = torch.vstack((zB, AC))
    zBAC = zBAC.roll(shifts=padding, dims=(-2, -1))
    zBAC = zBAC.to(z.device)
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
    laplacian = laplacian.numpy()
    laplacian = fftpack.dst(laplacian, type=1, axis=-2) / 2
    laplacian = fftpack.dst(laplacian, type=1, axis=-1) / 2
    laplacian = torch.from_numpy(laplacian)

    # compute Eigen Values
    u = torch.arange(1, H - 1)
    v = torch.arange(1, W - 1)
    u, v = torch.meshgrid(u, v, indexing="ij")
    laplacian = laplacian / (
        (2 * torch.cos(torch.pi * u / (H - 1)) - 2)
        + (2 * torch.cos(torch.pi * v / (W - 1)) - 2)
    )

    # compute Inverse Sine Transform
    laplacian = laplacian.numpy()
    laplacian = fftpack.idst(laplacian, type=1, axis=-2)
    laplacian = fftpack.idst(laplacian, type=1, axis=-1)
    laplacian = torch.from_numpy(laplacian)
    laplacian = laplacian / (laplacian.shape[0] + 1)
    laplacian = laplacian / (laplacian.shape[1] + 1)

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
