from torch import Tensor
from deepinv.physics.functional.multiplier import multiplier, multiplier_adjoint
from deepinv.physics.functional.convolution import conv2d, conv_transpose2d


def product_convolution(
    x: Tensor, w: Tensor, h: Tensor, padding: str = "valid"
) -> Tensor:
    r"""

    Product-convolution operator.
    Escande, P., & Weiss, P. (2017). Approximation of integral operators using product-convolution expansions. Journal of Mathematical Imaging and Vision, 58, 333-348.

    This forward operator performs

    .. math:: y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product, :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor x: Tensor of size (B, C, ...)
    :param torch.Tensor w: Tensor of size (K, b, c, ...)
    :param torch.Tensor h: Tensor of size (K, b, c, ...)
    :param padding: ( options = `valid`, `circular`, `replicate`, `reflect`. If `padding = 'valid'` the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.

    """

    K = w.shape[0]
    for k in range(K):
        if k == 0:
            result = conv2d(multiplier(x, w[k]), h[k], padding=padding)
        else:
            result += conv2d(multiplier(x, w[k]), h[k], padding=padding)

    return result


def product_convolution_adjoint(
    y: Tensor, w: Tensor, h: Tensor, padding: str = "valid"
) -> Tensor:
    r"""

    Product-convolution adjoint operator.

    :param torch.Tensor x: Tensor of size (B, C, ...)
    :param torch.Tensor w: Tensor of size (K, b, c, ...)
    :param torch.Tensor h: Tensor of size (K, b, c, ...)
    :param padding: ( options = `valid`, `circular`, `replicate`, `reflect`. If `padding = 'valid'` the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.
    """

    K = w.shape[0]
    for k in range(K):
        if k == 0:
            result = multiplier_adjoint(
                conv_transpose2d(y, h[k]), w[k], padding=padding
            )
        else:
            result += multiplier_adjoint(
                conv_transpose2d(y, h[k]), w[k], padding=padding
            )

    return result
