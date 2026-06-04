from torch import Tensor
from deepinv.physics.functional.multiplier import (
    multiplier,
    multiplier_adjoint,
)
import deepinv.physics.functional as dF
import torch


def product_convolution2d(
    x: Tensor, w: Tensor, h: Tensor, padding: str = "valid", use_fft: bool = False
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
    :param bool use_fft: whether to use FFT-based convolutions. If ``True``, it uses FFT-based convolutions which can be faster for large kernels.
    :return: :class:`torch.Tensor` the blurry image.
    """

    K = w.size(2)
    result = 0.0
    conv2d = dF.conv2d_fft if use_fft else dF.conv2d
    for k in range(K):
        result += conv2d(
            multiplier(x, w[:, :, k, ...]), h[:, :, k, ...], padding=padding
        )

    return result


def product_convolution2d_adjoint(
    y: Tensor, w: Tensor, h: Tensor, padding: str = "valid", use_fft: bool = False
) -> torch.Tensor:
    r"""

    Product-convolution adjoint operator in 2d.

    :param torch.Tensor x: Tensor of size (B, C, ...)
    :param torch.Tensor w: Tensor of size (b, c, K, ...)
    :param torch.Tensor h: Tensor of size (b, c, K, ...)
    :param padding: options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``.
        If `padding = 'valid'` the blurred output is smaller than the image (no padding),
        otherwise the blurred output has the same size as the image.
    :param bool use_fft: whether to use FFT-based convolutions. If ``True``, it uses FFT-based convolutions which can be faster for large kernels.
    """

    K = w.size(2)
    result = 0.0
    conv_transpose2d = dF.conv_transpose2d_fft if use_fft else dF.conv_transpose2d
    for k in range(K):
        result += multiplier_adjoint(
            conv_transpose2d(y, h[:, :, k, ...], padding=padding), w[:, :, k, ...]
        )

    return result
