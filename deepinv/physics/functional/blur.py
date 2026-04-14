from __future__ import annotations
import torch
from math import sqrt, pi


def gaussian_blur_nd(
    psf_size: tuple[int, ...] | None = None,
    sigma: int | float | tuple[float, ...] | torch.Tensor = (1.0, 1.0),
    angle: int | float | tuple[float, ...] | torch.Tensor = 0.0,
    batch_size: None | int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Creates a batch of N-dimensional anisotropic Gaussian kernels (1D, 2D, or 3D) with independent sigma ranges and rotation angles for each kernel.

    The dimension (1D, 2D, or 3D) of the kernel is determined by the length of the ``psf_size`` tuple, if ``None``, it is determined by the length of the ``sigma`` tuple. If both are provided, their dimensions must match.

    :param tuple[int, ...] psf_size: Shape of the point spread function for the batch, it can be a tuple specifying the size for each dimension (e.g., (7, 3) for 2D), or None to automatically determine the size based on sigma.
    :param int | float | tuple[float, ...] | torch.Tensor sigma: If a float or integer is provided, the kernel is isotropic with the specified sigma for all dimensions. The user can define an anisotropic kernel by passing a tuple of float. If a float, integer, or tuple is provided, the same kernel is repeated for the entire batch. If a tensor of shape (batch_size, dim) is provided, each kernel in the batch can have different sigma values.
    :param int | float | tuple[float, ...] | torch.Tensor angle: Rotation angle(s) for each kernel in the batch (degrees). For 2D kernels, this is a single angle of rotation in the plane. For 3D kernels, this can be a tuple of three angles (alpha, beta, gamma) representing rotations around the x, y, and z axes respectively. In 3D, if a single angle is provided, it is applied as a rotation around the z-axis. If a tuple of three angles is provided, they are applied as rotations around the x, y, and z axes in that order. If a tensor of shape (batch_size,) for 2D or (batch_size, 3) for 3D is provided, each kernel in the batch can have its own rotation angle(s).
    :param int batch_size: If sigma is an integer, float, or tuple, this parameter specifies how many kernels to generate in the batch. Ignored if sigma is a tensor.
    :param str device: Device to create the tensor on.

    :return: A tensor of batched N-dimensional Gaussian kernels of shape (batch_size, *psf_size).
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

    B = sigma.shape[0]  # batch size inferred from sigma shape

    # Rotation angles
    # ---------------
    # For 3D, angles is a list of three angles (alpha, beta, gamma)
    if dim == 2:
        if isinstance(angle, (int, float)):
            angle = torch.tensor(
                [angle] * B, device=device, dtype=dtype
            )  # Shape: (batch_size,)
        elif (
            isinstance(angle, torch.Tensor)
            and (angle.dim() == 1 or angle.dim() == 2 and angle.shape[1] == 1)
            and angle.shape[0] == B
        ):
            angle = angle.view(-1)  # Assume shape (batch_size,)
        else:
            raise ValueError(
                f"For 2D, angle must be a single value or a tensor of shape (batch_size,). Got angle.shape = {angle.shape}."
            )
    elif dim == 3:
        if isinstance(angle, (int, float)):
            angles = torch.tensor(
                [[angle, 0.0, 0.0]] * B, device=device, dtype=dtype
            )  # Shape: (batch_size, 3)
        elif isinstance(angle, (list, tuple)) and len(angle) == 3:
            angles = torch.tensor(
                [angle] * B, device=device, dtype=dtype
            )  # Shape: (batch_size, 3)
        elif (
            isinstance(angle, torch.Tensor)
            and angle.dim() == 2
            and angle.shape[1] == 3
            and angle.shape[0] == B
        ):
            angles = angle  # Shape: (batch_size, 3)
            # Check that angle is effectively of shape (batch_size, 3)
        else:
            raise ValueError(
                f"For 3D, angles must be a list of three angles (alpha, beta, gamma) or a tensor of shape (batch_size, 3). Got angle.shape = {angle.shape}."
            )

    angle = torch.deg2rad(
        angle
    )  # Convert angles from degrees to radians for rotation calculations

    # Create a grid for each dimension
    grids = []
    for d in range(dim):
        ax = torch.linspace(
            -((psf_size[d] - 1) / 2), (psf_size[d] - 1) / 2, psf_size[d], device=device
        )
        grids.append(ax)

    # Create a meshgrid for the coordinates using 'ij' indexing so that
    # the i-th grid corresponds to the i-th dimension in psf_size.
    mesh = torch.meshgrid(*[grids[d] for d in range(dim)], indexing="xy")
    coords = torch.stack(list(mesh), dim=-1)  # Shape: (*psf_size, dim)

    # Reshape coords for batch processing: (batch_size, *psf_size, dim)
    coords = coords.unsqueeze(0).expand(B, *psf_size, dim)

    # sigma is passed in (depth, height, width) order, but we want (x,y,z) order for the Gaussian formula, so we flip it
    sigma = torch.flip(sigma, dims=[-1])  # Shape: (batch_size, dim)

    if dim == 2:
        # Rotation matrix for 2D
        cos_theta = torch.cos(angle).view(-1)
        sin_theta = torch.sin(angle).view(-1)
        rot_mat = torch.zeros((B, 2, 2), device=device)
        rot_mat[:, 0, 0] = cos_theta.squeeze()
        rot_mat[:, 0, 1] = -sin_theta.squeeze()
        rot_mat[:, 1, 0] = sin_theta.squeeze()
        rot_mat[:, 1, 1] = cos_theta.squeeze()

        # Apply rotation: (batch_size, *psf_size, 2)
        coords = torch.einsum("bij,b...j->b...i", rot_mat, coords)

    elif dim == 3:

        # Rotation matrices for x, y, z axes
        alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]

        # Rotation around x-axis
        Rx = torch.zeros((B, 3, 3), device=device)
        Rx[:, 0, 0] = 1.0
        Rx[:, 1, 1] = torch.cos(alpha)
        Rx[:, 1, 2] = -torch.sin(alpha)
        Rx[:, 2, 1] = torch.sin(alpha)
        Rx[:, 2, 2] = torch.cos(alpha)

        # Rotation around y-axis
        Ry = torch.zeros((B, 3, 3), device=device)
        Ry[:, 0, 0] = torch.cos(beta)
        Ry[:, 0, 2] = torch.sin(beta)
        Ry[:, 1, 1] = 1.0
        Ry[:, 2, 0] = -torch.sin(beta)
        Ry[:, 2, 2] = torch.cos(beta)

        # Rotation around z-axis
        Rz = torch.zeros((B, 3, 3), device=device)
        Rz[:, 0, 0] = torch.cos(gamma)
        Rz[:, 0, 1] = -torch.sin(gamma)
        Rz[:, 1, 0] = torch.sin(gamma)
        Rz[:, 1, 1] = torch.cos(gamma)
        Rz[:, 2, 2] = 1.0

        # Combined rotation matrix: R = Rz @ Ry @ Rx
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))
        # Apply rotation: (batch_size, *psf_size, 3)
        coords = torch.einsum("bij,b...j->b...i", R, coords)

    # Compute the N-dimensional Gaussian
    kernel = torch.ones((B, *psf_size), device=device, dtype=dtype)
    for d in range(dim):
        kernel *= torch.exp(
            -0.5 * (coords[..., d] ** 2) / (sigma[:, d].view(-1, *[1] * dim) ** 2)
        ) / (sqrt(2 * pi) * sigma[:, d].view(-1, *[1] * dim))

    # Normalize each kernel
    kernel = kernel / torch.sum(kernel, dim=tuple(range(1, dim + 1)), keepdim=True)

    # Add single channel dimension to fit (B,C,*img_size) format
    kernel = kernel[:, None]

    return kernel
