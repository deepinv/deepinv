from __future__ import annotations
import torch.nn.functional as F
from torch import Tensor


def _prepare_rir_for_grouped(filter: Tensor, B: int, C: int) -> Tensor:
    r"""
    Broadcasts a batch of RIR filters of shape ``(b, c, K)`` with ``b in {1, B}`` and
    ``c in {1, C}`` to shape ``(B * C, 1, K)``, ready to be used with a grouped 1D
    convolution (``groups=B*C``).
    """
    if filter.dim() != 3:
        raise ValueError(
            f"filter must be a 3D tensor of shape (b, c, K), got {filter.dim()}D."
        )
    b, c, k = filter.size()
    if b not in (1, B):
        raise ValueError(f"Batch dimension of filter must be 1 or {B}, got {b}.")
    if c not in (1, C):
        raise ValueError(f"Channel dimension of filter must be 1 or {C}, got {c}.")

    if b == 1:
        filter = filter.expand(B, -1, -1)
    if c == 1:
        filter = filter.expand(-1, C, -1)

    return filter.reshape(B * C, 1, k).contiguous()


def causal_conv1d(x: Tensor, filter: Tensor) -> Tensor:
    r"""
    Causal 1D convolution of a batch of signals ``x`` with a batch of filters ``filter``.

    This performs, for every batch element and channel,

    .. math::

        y[t] = \sum_{k=0}^{K-1} h[k] \, x[t-k], \qquad x[t] = 0 \text{ for } t<0,

    i.e. a linear (non-circular) convolution truncated to the ``T`` first samples,
    which is the usual way of applying a causal FIR filter (e.g. a room impulse
    response) to a signal of finite length.

    The adjoint of this operation is :func:`deepinv.physics.functional.causal_conv1d_adjoint`.

    :param torch.Tensor x: signal of size ``(B, C, T)``.
    :param torch.Tensor filter: filter of size ``(b, c, K)`` where ``b`` can be either
        ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :return: :class:`torch.Tensor`: the filtered signal of size ``(B, C, T)``.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be a 3D tensor of shape (B, C, T), got {x.dim()}D.")

    B, C, T = x.size()
    K = filter.size(-1)
    filter = _prepare_rir_for_grouped(filter, B, C).flip(-1)

    x = F.pad(x.contiguous(), (K - 1, 0))
    x = x.reshape(1, B * C, -1)
    y = F.conv1d(x, filter, groups=B * C)
    return y.reshape(B, C, T)


def causal_conv1d_adjoint(y: Tensor, filter: Tensor) -> Tensor:
    r"""
    Adjoint of :func:`deepinv.physics.functional.causal_conv1d`.

    :param torch.Tensor y: signal of size ``(B, C, T)``.
    :param torch.Tensor filter: filter of size ``(b, c, K)`` where ``b`` can be either
        ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :return: :class:`torch.Tensor`: the output of size ``(B, C, T)``.
    """
    if y.dim() != 3:
        raise ValueError(f"y must be a 3D tensor of shape (B, C, T), got {y.dim()}D.")

    B, C, T = y.size()
    K = filter.size(-1)
    filter = _prepare_rir_for_grouped(filter, B, C)

    y = F.pad(y.contiguous(), (0, K - 1))
    y = y.reshape(1, B * C, -1)
    x = F.conv1d(y, filter, groups=B * C)
    return x.reshape(B, C, T)
