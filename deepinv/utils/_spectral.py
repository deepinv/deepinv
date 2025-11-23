"""Tools from spectral analysis"""

from __future__ import annotations

import torch


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


def liu_jia_pad(
    x: torch.Tensor, *, padding: tuple[int, int], alpha: int = 1
) -> torch.Tensor:
    """
    Liu-Jia Padding

    Real-world blurry images have decorrelated opposite boundaries unlike images synthetically blurred using circular filters. This make the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding :footcite:p:`liu2008reducing` is a pre-processing step that pads the input image to make it have smooth circular boundaries while preserving the original spectral content as much as possible.

    The implementation is adapted from `the one <https://github.com/cszn/USRNet>`_ featured in the work of :footcite:t:`zhang2020deep`.

    :param torch.Tensor x: Input tensor of shape (B, C, H, W)
    :param tuple(int, int) padding: Tuple specifying the amount of horizontal and vertical padding (pad_h, pad_w)
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

    def _compute_padding(x: torch.Tensor) -> torch.Tensor:
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
        laplacian = _dst1(laplacian, dim=-2, inverse=False)
        laplacian = _dst1(laplacian, dim=-1, inverse=False)

        # compute Eigen Values
        u = torch.arange(1, H - 1)
        v = torch.arange(1, W - 1)
        u, v = torch.meshgrid(u, v, indexing="ij")
        laplacian = laplacian / (
            (2 * torch.cos(torch.pi * u / (H - 1)) - 2)
            + (2 * torch.cos(torch.pi * v / (W - 1)) - 2)
        )

        # compute Inverse Sine Transform
        laplacian = _dst1(laplacian, dim=-2, inverse=True)
        laplacian = _dst1(laplacian, dim=-1, inverse=True)

        # put solution in inner points; outer points obtained from boundary image
        x[..., 1:-1, 1:-1] = laplacian
        return x

    if alpha == 1:
        A = _compute_padding(A)
        B = _compute_padding(B)
    else:
        A[..., alpha - 1 : -alpha + 1, :] = _compute_padding(
            A[..., alpha - 1 : -alpha + 1, :]
        )
        B[..., :, alpha - 1 : -alpha + 1] = _compute_padding(
            B[..., :, alpha - 1 : -alpha + 1]
        )

    C = torch.zeros(BC + (2 * alpha + padding_h, 2 * alpha + padding_w))
    C[..., :alpha, :] = B[..., -alpha:, :]
    C[..., -alpha:, :] = B[..., :alpha, :]
    C[..., :, :alpha] = A[..., :, -alpha:]
    C[..., :, -alpha:] = A[..., :, :alpha]

    if alpha == 1:
        C = _compute_padding(C)
    else:
        C[..., alpha - 1 : -alpha + 1, alpha - 1 : -alpha + 1] = _compute_padding(
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
