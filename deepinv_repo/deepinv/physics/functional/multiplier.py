import torch
from torch import Tensor


def multiplier(x: Tensor, mult: Tensor) -> torch.Tensor:
    r"""
    Implements diagonal matrices or multipliers :math:`x` and `mult`.
    The adjoint of this operation is :func:`deepinv.physics.functional.multiplier_adjoint`

    :param torch.Tensor x: Image of size `(B, C, ...)`.
    :param torch.Tensor filter: Filter of size `(b, c, ...)` where `b` can be either `1` or `B` and `c` can be either `1` or `C`.

    If `b = 1` or `c = 1`, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

    :return torch.Tensor : the output of the multiplier, same shape as :math:`x`
    """

    assert (
        x.dim() == mult.dim()
    ), "Input and filter must have the same number of dimension"
    return mult * x


def multiplier_adjoint(x: Tensor, mult: Tensor) -> torch.Tensor:
    r"""
    Implements the adjoint of diagonal matrices or multipliers :math:`x` and ``mult``.

    The adjoint of this operation is :func:`deepinv.physics.functional.multiplier`

    :param torch.Tensor x: Image of size ``(B, C, ...)``.
    :param torch.Tensor filter: Filter of size ``(b, c, ...)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

    :return torch.Tensor : the output of the multiplier, same shape as :math:`x`
    """

    assert (
        x.dim() == mult.dim()
    ), "Input and filter must have the same number of dimension"
    return torch.conj(mult) * x
