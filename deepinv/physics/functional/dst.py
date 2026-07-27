import torch


def dst1(
    x: torch.Tensor, *, dim: int = -1, inverse: bool = False, orthosf: bool = True
) -> torch.Tensor:
    r"""
    Compute the one-dimensional `discrete sine transform <https://en.wikipedia.org/wiki/Discrete_sine_transform>`_ of type I (DST-I) or its inverse (IDST-I).

    The DST-I of a vector :math:`x` of length :math:`N` is defined as

    .. math::

        \mathrm{DST-I}(x)_k = - \frac{1}{2} \Im(\mathrm{DFT}(y_(k+1))),

    where :math:`y` is the odd extension of :math:`x`. The IDST-I is defined as

    .. math::

        \mathrm{IDST-I}(x)_k = - \frac{1}{N + 1} \Im(\mathrm{DFT}(y_(k+1))).

    If ``orthosf=True``, it computes the orthogonal sign-flipped DST-I instead:

    .. math::

        \mathrm{OSFDST-I}(x)_k = \frac{1}{\sqrt{2N + 2}} \Im(\mathrm{DFT}(y_(k+1))).

    .. note::

        The orthogonal sign-flipped DST-I is its own inverse, hence when ``orthosf=True``, we have ``dst1(dst1(x)) = x`` and the parameter ``inverse`` has no effect.

    :param torch.Tensor x: Input tensor.
    :param int dim: Dimension along which to compute the transform. Default is -1 (the last dimension).
    :param bool inverse: If True, compute the inverse DST-I (IDST-I). If False (default), compute the DST-I. It has not effect when ``ortho=True``.
    :param bool orthosf: If True (default), compute the orthogonal sign-flipped DST-I, otherwise compute the standard DST-I.
    :return: (:class:`torch.Tensor`) The transformed tensor.
    """
    # Compute the DST-I or IDST-I from the DFT of y
    N = x.shape[dim]

    # Compute y the odd extension of x
    # y = (0, x_1, ..., x_N, 0, -x_N, ..., -x_1)
    shape_zeros = list(x.shape)
    shape_zeros[dim] = 1
    zeros = torch.zeros(shape_zeros, dtype=x.dtype, device=x.device)
    x_flipped = torch.flip(x, dims=[dim])
    y = torch.cat([zeros, x, zeros, -x_flipped], dim=dim)

    # Compute the DFT of y
    norm = "ortho" if orthosf else "backward"
    y = torch.fft.rfft(y, dim=dim, norm=norm)
    y = y.narrow(dim, 1, N)

    # Set the leading coefficient for forward and inverse transforms
    if not orthosf:
        c = -1 / 2 if not inverse else -1 / (N + 1)
    else:
        c = 1.0

    return c * y.imag
