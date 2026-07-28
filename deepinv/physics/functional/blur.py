from __future__ import annotations
import torch

from math import sqrt, pi

from .dst import dst1


def _resolve_batch_size(
    sigma: int | float | tuple[float, ...] | torch.Tensor,
    angle: int | float | tuple[float, ...] | torch.Tensor,
) -> int:
    if isinstance(sigma, (int, float, tuple, list)) and isinstance(
        angle, (int, float, tuple, list)
    ):
        batch_size = 1
    elif isinstance(sigma, torch.Tensor) and isinstance(angle, torch.Tensor):
        if sigma.shape[0] != angle.shape[0]:
            raise ValueError(
                f"Batch size inferred from sigma and angle must match. Got sigma.shape[0] = {sigma.shape[0]} and angle.shape[0] = {angle.shape[0]}."
            )
        batch_size = sigma.shape[0]
    elif isinstance(sigma, torch.Tensor):
        batch_size = sigma.shape[0]
    elif isinstance(angle, torch.Tensor):
        batch_size = angle.shape[0]
    else:
        batch_size = 1

    return batch_size


def _resolve_sigma(
    sigma: int | float | tuple[float, ...] | torch.Tensor,
    batch_size: int,
    device: str | torch.device,
    dtype: torch.dtype,
    dim: int,
) -> torch.Tensor:

    # Standard deviation components
    # -----------------------------

    # batch size of 1 with isotropic kernel
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor(
            [[sigma] * dim] * batch_size, device=device, dtype=dtype
        )  # Shape: (batch_size, dim)

    # batch size of 1 with potentially anisotropic kernel
    elif isinstance(sigma, (list, tuple)):
        if len(sigma) != dim:
            raise ValueError(
                f"Length of sigma tuple must match the number of dimensions {dim}."
            )
        sigma = torch.tensor(
            [sigma] * batch_size, device=device, dtype=dtype
        )  # Shape: (batch_size, dim)

    elif isinstance(sigma, torch.Tensor):
        if sigma.dim() != 2 or sigma.shape[1] != dim or sigma.shape[0] != batch_size:
            raise ValueError(
                f"If sigma is a tensor, it must have shape (batch_size, dim). Got sigma.shape = {sigma.shape}."
            )
        sigma = sigma.to(device=device, dtype=dtype)  # Ensure correct device and dtype

    else:
        raise ValueError(
            f"Invalid type for sigma. Expected int, float, tuple of floats, or tensor. Got {type(sigma)}."
        )

    # Add extra *psf_size dimensions for broadcasting in Gaussian formula
    # (batch_size, dim) -> (batch_size, dim, 1, 1, ...) with as many 1's as dimensions in psf_size
    sigma = sigma.view(batch_size, dim, *(1,) * dim)

    return sigma


def _resolve_angle(
    angle: int | float | tuple[float, ...] | torch.Tensor,
    batch_size: int,
    device: str | torch.device,
    dtype: torch.dtype,
    dim: int,
) -> torch.Tensor:

    # Rotation angles
    # ---------------
    if dim == 2:
        if isinstance(angle, (int, float)):
            angle = torch.tensor(
                [angle] * batch_size, device=device, dtype=dtype
            )  # Shape: (batch_size,)
        elif (
            isinstance(angle, torch.Tensor)
            and (angle.dim() == 1 or angle.dim() == 2 and angle.shape[1] == 1)
            and angle.shape[0] == batch_size
        ):
            angle = angle.to(device=device, dtype=dtype)  # Assume shape (batch_size,)
        else:
            raise ValueError(
                f"For 2D, angle must be a single value or a tensor of shape (batch_size,). Got angle.shape = {angle.shape}."
            )

    # For 3D, angles is a list of three angles (alpha, beta, gamma)
    elif dim == 3:
        if isinstance(angle, (int, float)):
            angle = torch.tensor(
                [[angle, 0.0, 0.0]] * batch_size, device=device, dtype=dtype
            )  # Shape: (batch_size, 3)
        elif isinstance(angle, (list, tuple)) and len(angle) == 3:
            angle = torch.tensor(
                [angle] * batch_size, device=device, dtype=dtype
            )  # Shape: (batch_size, 3)
        elif (
            isinstance(angle, torch.Tensor)
            and angle.dim() == 2
            and angle.shape[1] == 3
            and angle.shape[0] == batch_size
        ):
            angle = angle.to(
                device=device, dtype=dtype
            )  # Ensure correct device and dtype
        else:
            raise ValueError(
                f"For 3D, angles must be a list of three angles (alpha, beta, gamma) or a tensor of shape (batch_size, 3). Got angle.shape = {angle.shape}."
            )

    if dim in (2, 3):
        angle = torch.deg2rad(
            angle
        )  # Convert angles from degrees to radians for rotation calculations

    return angle


def gaussian_blur(
    psf_size: tuple[int, ...] | None = None,
    sigma: int | float | tuple[float, ...] | torch.Tensor = (1.0, 1.0),
    angle: int | float | tuple[float, ...] | torch.Tensor = 0.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Creates a batch of N-dimensional anisotropic Gaussian kernels (1D, 2D, or 3D) with independent sigma ranges and rotation angles for each kernel.

    The dimension (1D, 2D, or 3D) of the kernel is determined by the length of the ``psf_size`` tuple, if ``None``, it is determined by the length of the ``sigma`` tuple. If both are provided, their dimensions must match. Batch size is determined by the shape of ``sigma`` or ``angle``, if both are provided, their shapes must be compatible. If both are scalars (int, float or a tuple), the batch size is 1 and the same kernel is repeated for the entire batch. If ``sigma`` is a tensor of shape (batch_size, dim), then each kernel in the batch can have different sigma values.

    :param tuple[int, ...] psf_size: Shape of the point spread function for the batch, it can be a tuple specifying the size for each dimension (e.g., (7, 3) for 2D), or None to automatically determine the size based on sigma.
    :param int | float | tuple[float, ...] | torch.Tensor sigma: If a float or integer is provided, the kernel is isotropic with the specified sigma for all dimensions. The user can define an anisotropic kernel by passing a tuple of float. If a float, integer, or tuple is provided, the same kernel is repeated for the entire batch. If a tensor of shape (batch_size, dim) is provided, each kernel in the batch can have different sigma values.
    :param int | float | tuple[float, ...] | torch.Tensor angle: Rotation angle(s) for each kernel in the batch (degrees). For 2D kernels, this is a single angle of rotation in the plane. For 3D kernels, this can be a tuple of three angles (gamma, beta, alpha) representing rotations around the x, y, and z axes respectively. In 3D, if a single angle is provided, it is applied as a rotation around the z-axis. If a tuple of three angles is provided, they are applied as rotations around the x, y, and z axes in that order. If a tensor of shape (batch_size,) for 2D or (batch_size, 3) for 3D is provided, each kernel in the batch can have its own rotation angle(s).
    :param str device: Device to create the tensor on.

    :return: A tensor of batched N-dimensional Gaussian kernels of shape ``(batch_size, *psf_size)``.
    """

    if psf_size is None:
        dim = 1 if isinstance(sigma, (int, float)) else len(sigma)

        s = sigma if isinstance(sigma, (int, float)) else max(sigma)
        c = int(s / 0.3 + 1)
        psf_size = (2 * c + 1,) * dim

    else:
        dim = len(psf_size)

    if dim not in {1, 2, 3}:
        raise ValueError("Only 1D, 2D, and 3D kernels are supported.")

    # Resolve batch size from sigma and angle inputs
    batch_size = _resolve_batch_size(sigma, angle)

    # Format sigma into tensor of shape (batch_size, dim)
    sigma = _resolve_sigma(
        sigma, batch_size=batch_size, device=device, dtype=dtype, dim=dim
    )
    # Format angle into tensor of shape (batch_size,) for 2D angle or (batch_size, 3) for 3D angle
    angle = _resolve_angle(
        angle, batch_size=batch_size, device=device, dtype=dtype, dim=dim
    )

    # Create a grid for each dimension
    grids = []
    for d in range(dim):
        ax = torch.linspace(
            -((psf_size[d] - 1) / 2),
            (psf_size[d] - 1) / 2,
            psf_size[d],
            device=device,
            dtype=dtype,
        )
        grids.append(ax)

    # Create a meshgrid for the coordinates using 'ij' indexing so that
    # the i-th grid corresponds to the i-th dimension in psf_size.
    # mesh[-1] will correspond to the x-coordinates, mesh[-2] to the y-coordinates, and mesh[-3] to the z-coordinates (if 3D).
    mesh = torch.meshgrid(*grids, indexing="ij")

    # coords will have shape (*psf_size, dim) where the last dimension corresponds
    # to the coordinates values in (x,y,z) order.
    coords = torch.stack(list(mesh)[::-1], dim=-1)  # Shape: (*psf_size, dim)

    # Reshape coords for batch processing: (batch_size, *psf_size, dim)
    coords = coords.unsqueeze(0).expand(batch_size, *psf_size, dim)

    # sigma is passed in (depth, height, width) order, but we want (x,y,z) order for the Gaussian formula, so we flip it
    sigma = torch.flip(sigma, dims=(1,))  # Shape: (batch_size, dim, *psf_dims)

    if dim == 2:
        # Rotation matrix for 2D
        cos_theta = torch.cos(angle).view(-1)
        sin_theta = torch.sin(angle).view(-1)
        rot_mat = torch.stack(
            [cos_theta, -sin_theta, sin_theta, cos_theta], dim=1
        ).view(batch_size, 2, 2)

        # Apply rotation: (batch_size, *psf_size, 2)
        coords = torch.einsum("bij,b...j->b...i", rot_mat, coords)

    elif dim == 3:

        # Rotation matrices for x, y, z axes
        # Taking https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations as reference
        gamma, beta, alpha = angle[:, 0], angle[:, 1], angle[:, 2]

        # Rotation around x-axis
        R = torch.zeros((batch_size, 3, 3), device=device)
        R[:, 0, 0] = torch.cos(alpha) * torch.cos(beta)
        R[:, 0, 1] = torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma) - torch.sin(
            alpha
        ) * torch.cos(gamma)
        R[:, 0, 2] = torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma) + torch.sin(
            alpha
        ) * torch.sin(gamma)

        R[:, 1, 0] = torch.sin(alpha) * torch.cos(beta)
        R[:, 1, 1] = torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.cos(
            alpha
        ) * torch.cos(gamma)
        R[:, 1, 2] = torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.cos(
            alpha
        ) * torch.sin(gamma)

        R[:, 2, 0] = -torch.sin(beta)
        R[:, 2, 1] = torch.cos(beta) * torch.sin(gamma)
        R[:, 2, 2] = torch.cos(beta) * torch.cos(gamma)
        # Apply rotation: (batch_size, *psf_size, 3)
        coords = torch.einsum("bij,b...j->b...i", R, coords)

    # Compute the N-dimensional Gaussian
    kernel = torch.ones((batch_size, *psf_size), device=device, dtype=dtype)
    # coords = coords.permute(0, *reversed(tuple(range(1, dim + 1))), -1)
    for d in range(dim):
        kernel *= torch.exp(-0.5 * (coords[..., d] ** 2) / (sigma[:, d] ** 2)) / (
            sqrt(2 * pi) * sigma[:, d]
        )

    # Normalize each kernel
    kernel = kernel / torch.sum(kernel, dim=tuple(range(1, dim + 1)), keepdim=True)

    # Add single channel dimension to fit (B,C,*img_size) format
    kernel = kernel[:, None]

    return kernel


def kaiser_window(
    beta: float, length: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Return the Kaiser window of length `length` and shape parameter `beta`."""
    if beta < 0:
        raise ValueError("beta must be greater than 0")
    if length < 1:
        raise ValueError("length must be greater than 0")
    if length == 1:
        return torch.tensor([1.0])
    half = (length - 1) / 2
    n = torch.arange(length, device=device)
    beta = torch.tensor(beta, device=device)
    return torch.i0(beta * torch.sqrt(1 - ((n - half) / half) ** 2)) / torch.i0(beta)


def sinc_filter(
    factor: float | torch.Tensor = 2,
    length: int = 11,
    windowed: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    r"""
    Anti-aliasing sinc filter, optionally multiplied by a Kaiser window.

    The kaiser window parameter is computed as follows:

    .. math::

        A = 2.285 \cdot (L - 1) \cdot 3.14 \cdot \Delta f + 7.95

    where :math:`\Delta f = 2 (2 - \sqrt{2}) / \text{factor}`. Then, the beta parameter is computed as:

    .. math::

            \beta = \begin{cases}
                0 & \text{if } A \leq 21 \\
                0.5842 \cdot (A - 21)^{0.4} + 0.07886 \cdot (A - 21) & \text{if } 21 < A \leq 50 \\
                0.1102 \cdot (A - 8.7) & \text{otherwise}
            \end{cases}

    :param float, torch.Tensor factor: Downsampling factor. If Tensor, can only have one element.
    :param int length: Length of the filter.
    :param bool windowed: Whether to multiply by Kaiser window.
    :param torch.device, str device: device to put the filter on (cpu or cuda)
    """
    if isinstance(factor, torch.Tensor):
        factor = factor.cpu().item()

    deltaf = 2 * (2 - 1.4142136) / factor

    n = torch.arange(length, device=device) - (length - 1) / 2
    filter = torch.sinc(n / factor)

    if windowed:
        A = 2.285 * (length - 1) * 3.14159 * deltaf + 7.95
        if A <= 21:
            beta = 0
        elif A <= 50:
            beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
        else:
            beta = 0.1102 * (A - 8.7)

        filter = filter * kaiser_window(beta, length, device=device)

    filter = filter.unsqueeze(0)
    filter = filter * filter.T
    filter = filter.unsqueeze(0).unsqueeze(0)
    filter = filter / filter.sum()
    return filter


def bilinear_filter(
    factor: int = 2, device: torch.device | str = "cpu"
) -> torch.Tensor:
    r"""
    Bilinear filter.

    It has size (2*factor, 2*factor) and is defined as

    .. math::

            w(x, y) = \begin{cases}
                (1 - |x|) \cdot (1 - |y|) & \text{if } |x| \leq 1 \text{ and } |y| \leq 1 \\
                0 & \text{otherwise}
            \end{cases}

    for :math:`x, y \in {-\text{factor} + 0.5, -\text{factor} + 0.5 + 1/\text{factor}, \ldots, \text{factor} - 0.5}`.

    :param int factor: downsampling factor
    :param torch.device, str device: device to put the filter on (cpu or cuda)
    """
    if isinstance(factor, torch.Tensor):
        factor = factor.cpu().item()
    x = torch.arange(start=-factor + 0.5, end=factor, step=1, device=device) / factor
    w = 1 - x.abs()
    w = torch.outer(w, w)
    w = w / torch.sum(w)
    return w.unsqueeze(0).unsqueeze(0)


def _biharmonic_inpainting(x: torch.Tensor) -> None:
    """
    Biharmonic inpainting

    The biharmonic inpainting of an image :math:`x` smooth at the image
    boundary and zero inside the boundary is a smooth image :math:`y`
    such that :math:`x` and :math:`y` are equal on the boundary and
    :math:`\nabla^2 y = 0` inside the boundary, where :math:`\nabla^2` is the
    Laplacian operator :footcite:p:`damelin2018surface`.

    In the discrete setting, the boundary is one pixel thick and the Laplacian
    operator is approximated by the 5-point stencil. It can be diagonalized in
    the discrete sine transform of type I (DST-I) basis, which gives the
    formula:

    .. math::

        y = - \mathcal{S}^{-1} \mathrm{diag}(d^{-1}) \mathcal{S} \nabla^2 x + x

    where :math:`\mathcal{S}` and :math:`\mathcal{S}^{-1}` are the DST-I and
    its inverse, :math:`d` denotes the eigenvalues of the 5-point Laplacian
    operator :math:`\nabla^2`. The eigenvalues :math:`d` are given by

    .. math::

        d_{k,l} = 2 (\cos(\pi k / (H - 1)) + \cos(\pi l / (W - 1)) - 2) \quad k = 0, \ldots, H - 1, \quad l = 0, \ldots, W - 1

    .. note::

        This function is used in the Liu-Jia padding :func:`deepinv.physics.functional.liu_jia_pad`.

    .. warning::

        This function modifies the input image in-place and it assumes that it
        is zero outside the one pixel thick boundary.
    """
    # Compute the 5-point Laplacian of x
    laplacian = (
        x[..., 1:-1, 2:]
        + x[..., 1:-1, :-2]
        + x[..., 2:, 1:-1]
        + x[..., :-2, 1:-1]
        - 4 * x[..., 1:-1, 1:-1]
    )

    # Compute the DST-I of the Laplacian
    spec = dst1(laplacian, dim=(-2, -1), inverse=False, orthosf=False)

    # Compute the eigenvalues of the 5-point Laplacian operator
    H, W = x.shape[-2:]
    f_h = torch.arange(1, H - 1, device=x.device, dtype=x.dtype)
    f_w = torch.arange(1, W - 1, device=x.device, dtype=x.dtype)
    f_h, f_w = torch.meshgrid(f_h, f_w, indexing="ij")
    d = (
        2 * torch.cos(torch.pi * f_h / (H - 1))
        + 2 * torch.cos(torch.pi * f_w / (W - 1)) - 4
    )

    # Multiply the DST-I by the negative inverse of the eigenvalues of the
    # 5-point Laplacian operator
    z = - spec / d

    # Compute the inverse DST-I of the result
    z = dst1(z, dim=(-2, -1), inverse=True, orthosf=False)

    # Add the result in-place with the original boundary to get the output
    # y = z + x
    x[..., 1:-1, 1:-1] = z


def liu_jia_pad(
    x: torch.Tensor, *, padding: tuple[int, int], alpha: int = 1
) -> torch.Tensor:
    """
    Liu-Jia Padding

    Real-world blurry images have decorrelated opposite boundaries unlike images synthetically blurred using circular filters. This make the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding :footcite:p:`liu2008reducing` is a pre-processing step that pads the input image to make it have smooth circular boundaries while preserving the original spectral content as much as possible.

    The implementation is adapted from `the one <https://github.com/cszn/USRNet>`_ featured in the work of :footcite:t:`zhang2020deep`.

    The padded tensor has shape :math:`(B, C, H + 2 * \text{pad}_h, W + 2 * \text{pad}_w)` where :math:`\text{pad}_h` and :math:`\text{pad}_w` are the vertical and horizontal padding respectively.

    :param torch.Tensor x: Input image of shape (B, C, H, W)
    :param tuple(int, int) padding: Horizontal and vertical padding (px)
    :param int alpha: Border width (px). Default: 1
    :return: (:class:`torch.Tensor`) Padded image
    """
    if x.ndim != 4:  # pragma: no cover
        raise ValueError("Input tensor must be 4-dimensional (B, C, H, W)")

    padding_h = 2 * padding[0]
    padding_w = 2 * padding[1]

    shape = x.shape
    BC = tuple(x.shape[:-2])
    H, W = x.shape[-2:]

    # The padded image is written (up to a shift) as
    # [ x, B ]
    # [ A, C ]
    # and A, B and C are computed separately

    A_shape = BC + (2 * alpha + padding_h, W)
    B_shape = BC + (H, 2 * alpha + padding_w)
    C_shape = BC + (2 * alpha + padding_h, 2 * alpha + padding_w)

    A = torch.zeros(A_shape, device=x.device, dtype=x.dtype)
    B = torch.zeros(B_shape, device=x.device, dtype=x.dtype)
    C = torch.zeros(C_shape, device=x.device, dtype=x.dtype)

    a = torch.linspace(0, 1, padding_h, device=x.device, dtype=x.dtype)
    b = torch.linspace(0, 1, padding_w, device=x.device, dtype=x.dtype)

    a = a.view((1,) * len(BC) + a.shape)
    b = b.view((1,) * len(BC) + b.shape)

    A[..., :alpha, :] = x[..., -alpha:, :]
    A[..., -alpha:, :] = x[..., :alpha, :]
    A[..., alpha:-alpha, 0] = (1 - a) * A[..., alpha - 1, 0, None] + a * A[
        ..., -alpha, 0, None
    ]
    A[..., alpha:-alpha, -1] = (1 - a) * A[..., alpha - 1, -1, None] + a * A[
        ..., -alpha, -1, None
    ]

    B[..., :, :alpha] = x[..., :, -alpha:]
    B[..., :, -alpha:] = x[..., :, :alpha]
    B[..., 0, alpha:-alpha] = (1 - b) * B[..., 0, alpha - 1, None] + b * B[
        ..., 0, -alpha, None
    ]
    B[..., -1, alpha:-alpha] = (1 - b) * B[..., -1, alpha - 1, None] + b * B[
        ..., -1, -alpha, None
    ]

    # Compute the values inside the boundaries
    stop_h = A.shape[-2] - (alpha - 1)
    stop_w = B.shape[-1] - (alpha - 1)

    _biharmonic_inpainting(A[..., alpha - 1 : stop_h, :])
    _biharmonic_inpainting(B[..., :, alpha - 1 : stop_w])

    C[..., :alpha, :] = B[..., -alpha:, :]
    C[..., -alpha:, :] = B[..., :alpha, :]
    C[..., :, :alpha] = A[..., :, -alpha:]
    C[..., :, -alpha:] = A[..., :, :alpha]

    _biharmonic_inpainting(C[..., alpha - 1 : stop_h, alpha - 1 : stop_w])

    # Crop the boundary
    A = A[..., alpha - 1 : -alpha - 1, :]
    B = B[..., :, alpha:-alpha]
    C = C[..., alpha:-alpha, alpha:-alpha]

    # Concatenate A, B, C and x to form the padded image
    # [ x, B ]
    # [ A, C ]
    H, W = x.shape[-2:]
    Hp, Wp = A.shape[-2], B.shape[-1]
    zBAC = torch.empty(BC + (H + Hp, W + Wp), device=x.device, dtype=x.dtype)
    zBAC[..., :H, :W] = x
    zBAC[..., :H, W:] = B
    zBAC[..., H:, :W] = A
    zBAC[..., H:, W:] = C

    # Realign the padded image so the original image is centered
    zBAC = zBAC.roll(shifts=padding, dims=(-2, -1))

    return zBAC


def bicubic_filter(factor: int = 2, device: torch.device | str = "cpu") -> torch.Tensor:
    r"""
    Bicubic filter.

    It has size (4*factor, 4*factor) and is defined as

    .. math::

        w(x, y) = \begin{cases}
            (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{if } |x| \leq 1 \\
            a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{if } 1 < |x| < 2 \\
            0 & \text{otherwise}
        \end{cases}

    for :math:`x, y \in {-2\text{factor} + 0.5, -2\text{factor} + 0.5 + 1/\text{factor}, \ldots, 2\text{factor} - 0.5}`.

    :param int factor: downsampling factor
    :param torch.device, str device: device to put the filter on (cpu or cuda)
    """
    if isinstance(factor, torch.Tensor):
        factor = factor.cpu().item()
    x = (
        torch.arange(start=-2 * factor + 0.5, end=2 * factor, step=1, device=device)
        / factor
    )
    a = -0.5
    x = x.abs()
    w = ((a + 2) * x.pow(3) - (a + 3) * x.pow(2) + 1) * (x <= 1)
    w += (a * x.pow(3) - 5 * a * x.pow(2) + 8 * a * x - 4 * a) * (x > 1) * (x < 2)
    w = torch.outer(w, w)
    w = w / torch.sum(w)
    return w.unsqueeze(0).unsqueeze(0)
