from __future__ import annotations

import torch
import torch.nn.functional as F


def rotate_grid(
    n_x: int,
    theta: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""
    Build sampling grids that rotate square ``n_x x n_x`` planes, on ``theta``'s device.

    :param int n_x: transaxial size. Requires a square isotropic grid
    :param torch.Tensor theta: ``(L,)`` rotation angles (rad)
    :param torch.dtype dtype: output dtype, defaults to the global default.
    :return: ``(L, n_x, n_x, 2)`` grid for :func:`torch.nn.functional.grid_sample`
    """
    theta = theta.double()
    c, s = torch.cos(theta), torch.sin(theta)
    mat = torch.zeros(theta.numel(), 2, 3, dtype=torch.float64, device=theta.device)
    mat[:, 0, 0], mat[:, 1, 1] = c, c
    mat[:, 0, 1], mat[:, 1, 0] = -s, s  # third column stays 0: rotate about centre
    return F.affine_grid(
        mat.to(dtype or torch.get_default_dtype()),
        (theta.numel(), 1, n_x, n_x),
        align_corners=False,
    )


def rotate(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    r"""
    Rotate transaxial planes.

    :param torch.Tensor x: ``(N, C, n_x, n_y)``. same grid applied to every channel C
    :param torch.Tensor grid: ``(N, n_x, n_x, 2)``
    """
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


class _RotateAdjoint(torch.autograd.Function):
    r"""
    transpose :math:`R^{\top}` of :func:`rotate (not rotating by :math:`-\theta`)

    keeps ``A_adjoint`` differentiable in a single pass
    """

    @staticmethod
    def forward(ctx, y: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(grid)
        gx, _ = torch.ops.aten.grid_sampler_2d_backward(
            y.contiguous(),  # shape == input of rotate
            torch.zeros_like(y),
            grid,
            0,  # interpolation_mode: bilinear
            0,  # padding_mode: zeros
            False,  # align_corners
            [True, False],  # gradients (input, grid)
        )
        return gx

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> tuple[torch.Tensor, None]:
        (grid,) = ctx.saved_tensors
        return rotate(g, grid), None


def rotate_adjoint(y: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    r"""exact transpose of :func:`rotate`

    :param torch.Tensor y: ``(N, C, n_x, n_x)``
    :param torch.Tensor grid: ``(N, n_x, n_x, 2)`` from :func:`rotate_grid`
    :return: ``(N, C, n_x, n_y)``, the shape of :func:`rotate`'s input
    """
    return _RotateAdjoint.apply(y, grid)
