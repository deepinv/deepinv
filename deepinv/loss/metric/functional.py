from typing import Optional
import torch


def norm(a):
    """Computes the L2 norm i.e. root sum of squares"""
    return a.pow(2).sum(dim=(-1, -2), keepdim=True).sqrt()


def cal_psnr(
    a: torch.Tensor,
    b: torch.Tensor,
    max_pixel: float = 1.0,
    min_pixel: float = 0.0,
):
    r"""
    Computes the peak signal-to-noise ratio (PSNR).

    :param torch.Tensor a: tensor estimate
    :param torch.Tensor b: tensor reference
    :param float max_pixel: maximum pixel value
    :param float min_pixel: minimum pixel value
    """
    with torch.no_grad():
        psnr = -10.0 * torch.log10(cal_mse(a, b) / (max_pixel - min_pixel) ** 2 + 1e-8)
    return psnr


def cal_mse(a, b):
    """Computes the mean squared error (MSE)"""
    return (a - b).pow(2).mean(dim=tuple(range(1, a.ndim)), keepdim=False)


def cal_mae(a, b):
    """Computes the mean absolute error (MAE)"""
    return (a - b).abs().mean(dim=tuple(range(1, a.ndim)), keepdim=False)


def complex_abs(data: Optional[torch.Tensor], dim=1, keepdim=True):
    """
    Compute the absolute value of a complex valued input tensor.

    If data has length 2 in the channel dimension given by dim, assumes this represents Re and Im parts.
    If data is a ``torch.complex`` dtype, takes absolute directly.

    :param torch.Tensor data: A complex valued tensor.
    :param int dim: complex dimension
    :param bool keepdim: keep complex dimension after abs
    """
    if data is None:
        return data

    if data.is_complex():
        return torch.abs(data)
    else:
        assert data.size(dim) == 2
        return (data**2).sum(dim=dim, keepdim=keepdim).sqrt()
