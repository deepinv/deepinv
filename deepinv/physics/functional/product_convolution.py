from torch import Tensor
from deepinv.physics.functional.multiplier import (
    multiplier,
    multiplier_adjoint,
)
from deepinv.physics.functional.convolution import conv2d, conv_transpose2d
import torch


def product_convolution2d(
    x: Tensor, w: Tensor, h: Tensor, padding: str = "valid"
) -> torch.Tensor:
    r"""

    Product-convolution operator in 2d. Details available in the paper :footcite:t:`escande2017approximation`.

    This forward operator performs

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product, :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor x: Tensor of size :math:`(B, C, H, W)`
    :param torch.Tensor w: Tensor of size :math:`(b, c, K, H, W)`. :math:`b \in \{1, B\}` and :math:`c \in \{1, C\}`
    :param torch.Tensor h: Tensor of size :math:`(b, c, K, h, w)`. :math:`b \in \{1, B\}` and :math:`c \in \{1, C\}`, :math:`h\leq H` and :math:`w\leq W`.
    :param padding: ( options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'`` or ``'constant'``). If `padding = `'valid'` the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.

    :return: :class:`torch.Tensor` the blurry image.
    """

    K = w.size(2)
    result = 0.0
    for k in range(K):
        result += conv2d(
            multiplier(x, w[:, :, k, ...]), h[:, :, k, ...], padding=padding
        )

    return result


def product_convolution2d_adjoint(
    y: Tensor, w: Tensor, h: Tensor, padding: str = "valid"
) -> torch.Tensor:
    r"""

    Product-convolution adjoint operator in 2d.

    :param torch.Tensor x: Tensor of size (B, C, ...)
    :param torch.Tensor w: Tensor of size (b, c, K, ...)
    :param torch.Tensor h: Tensor of size (b, c, K, ...)
    :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
        If `padding = 'valid'` the blurred output is smaller than the image (no padding),
        otherwise the blurred output has the same size as the image.
    """

    K = w.size(2)
    result = 0.0
    for k in range(K):
        result += multiplier_adjoint(
            conv_transpose2d(y, h[:, :, k, ...], padding=padding), w[:, :, k, ...]
        )

    return result


def get_psf_pconv2d_eigen(h, w, position):
    r"""
    Get the PSF at the given position of the :meth:`deepinv.physics.functional.product_convolution2d` function.
    :param torch.Tensor w: Tensor of size (B, C, K, H, W).
    :param torch.Tensor h: Tensor of size (B, C, K, h, w).
    :param torch.Tensor position: Position of the PSF, a Tensor of size (B, n_position, 2)

    :return torch.Tensor: PSF at position of shape (B, C, n_position, psf_size, psf_size)
    """
    batch_index = torch.arange(w.size(0), dtype=torch.long, device=w.device)
    position_h = position[..., 0:1]
    position_w = position[..., 1:2]
    w_selected = (
        w[
            batch_index[:, None, None],
            ...,
            position_h,
            position_w,
        ]
        .squeeze(2)
        .transpose(1, 2)
    )
    return torch.sum(h[:, :, None, ...] * w_selected[..., None, None], dim=3).flip(
        (-1, -2)
    )
