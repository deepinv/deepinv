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

    :param torch.Tensor x: Tensor of size (B, C, H, W)
    :param torch.Tensor w: Tensor of size (b, c, K, H, W). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (b, c, K, h, w). b in {1, B} and c in {1, C}, h<=H and w<=W
    :param padding: ( options = `valid`, `circular`, `replicate`, `reflect`. If `padding = 'valid'` the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.

    :return: torch.Tensor y
    :rtype: tuple
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
