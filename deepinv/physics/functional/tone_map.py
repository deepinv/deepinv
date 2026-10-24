import math

import torch


def srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    r"""
    sRGB opto-electronic transfer function (OETF).

    Encodes linear-light sRGB values into non-linear sRGB values, following
    the sRGB standard :footcite:t:`iec1999srgb`:

    .. math::

        y = \begin{cases}
            12.92\, x & \text{if } x \leq 0.0031308 \\
            1.055\, x^{1/2.4} - 0.055 & \text{else}
        \end{cases}

    This is the encoding step applied by a camera ISP when saving linear
    sensor values to an sRGB image. It is the exact inverse of
    :func:`deepinv.physics.functional.srgb_eotf` on :math:`[0, 1]`.

    :param torch.Tensor x: input tensor of linear-light values.
    :return: (:class:`torch.Tensor`) non-linear (encoded) sRGB values.
    """
    # clamp the power branch to its domain, so that its (unused) gradient stays
    # finite on the linear branch and does not turn into NaN in torch.where
    xc = torch.clamp(x, min=0.0031308)
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * torch.pow(xc, 1.0 / 2.4) - 0.055,
    )


def srgb_eotf(y: torch.Tensor) -> torch.Tensor:
    r"""
    sRGB electro-optical transfer function (EOTF).

    Decodes non-linear sRGB values back into linear-light values, following
    the sRGB standard :footcite:t:`iec1999srgb`:

    .. math::

        x = \begin{cases}
            y / 12.92 & \text{if } y \leq 0.04045 \\
            \left(\frac{y + 0.055}{1.055}\right)^{2.4} & \text{else}
        \end{cases}

    This is the exact inverse of :func:`deepinv.physics.functional.srgb_oetf`
    on :math:`[0, 1]`.

    :param torch.Tensor y: input tensor of non-linear (encoded) sRGB values.
    :return: (:class:`torch.Tensor`) linear-light values.

    """
    yc = torch.clamp(y, min=0.0)
    return torch.where(
        y <= 0.04045,
        y / 12.92,
        torch.pow((yc + 0.055) / 1.055, 2.4),
    )


def gamma_tone(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    r"""
    Gamma tone mapping.

    Encodes linear values with a power law :math:`y = x^{1/\gamma}`. This is a
    simplified model of the encoding curves used by e.g. Adobe RGB or video
    gamma standards.

    .. math::

        y = x^{1/\gamma}

    :param torch.Tensor x: input tensor of linear values.
    :param float gamma: encoding gamma :math:`\gamma > 0`. Default: ``2.2``.
    :return: (:class:`torch.Tensor`) tone mapped values.
    """
    # the power has an infinite gradient at 0: evaluate it on positive values only
    pos = x > 0
    return torch.where(pos, torch.pow(torch.where(pos, x, 1.0), 1.0 / gamma), 0.0)


def gamma_tone_inverse(y: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    r"""
    Inverse gamma tone mapping :math:`x = y^{\gamma}`.

    This is the exact inverse of :func:`deepinv.physics.functional.gamma_tone`
    on :math:`[0, \infty)`.

    :param torch.Tensor y: input tensor of tone mapped values.
    :param float gamma: encoding gamma :math:`\gamma > 0`. Default: ``2.2``.
    :return: (:class:`torch.Tensor`) linear values.
    """
    return torch.pow(torch.clamp(y, min=0.0), gamma)


def log_tone(x: torch.Tensor, L: float = 100.0) -> torch.Tensor:
    r"""
    Logarithmic tone mapping.

    Maps linear values to :math:`[0, 1]` with a logarithmic curve, a
    simplified version of the adaptive logarithmic mapping of
    :footcite:t:`drago2003adaptive`:

    .. math::

        y = \frac{\log(1 + x)}{\log(1 + L)}

    where :math:`L` is the maximum linear value that is mapped to 1. Larger
    values of :math:`L` yield more aggressive highlight compression.

    :param torch.Tensor x: input tensor of linear values.
    :param float L: maximum linear value mapped to 1, :math:`L > 0`. Default: ``100.0``.
    :return: (:class:`torch.Tensor`) tone mapped values.
    """
    return torch.log1p(torch.clamp(x, min=0.0)) / math.log1p(L)


def log_tone_inverse(y: torch.Tensor, L: float = 100.0) -> torch.Tensor:
    r"""
    Inverse logarithmic tone mapping :math:`x = \exp\left(y \log(1 + L)\right) - 1`.

    This is the exact inverse of :func:`deepinv.physics.functional.log_tone`
    on :math:`[0, \infty)`.

    :param torch.Tensor y: input tensor of tone mapped values.
    :param float L: maximum linear value mapped to 1, :math:`L > 0`. Default: ``100.0``.
    :return: (:class:`torch.Tensor`) linear values.
    """
    return torch.expm1(torch.clamp(y, min=0.0) * math.log1p(L))


def reinhard_tone(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    r"""
    Reinhard tone mapping.

    Maps linear values to :math:`[0, 1)` with the curve of
    :footcite:t:`reinhard2002photographic`:

    .. math::

        y = \frac{\alpha x}{1 + \alpha x}

    where :math:`\alpha > 0` is an exposure-like parameter: larger values of
    :math:`\alpha` make the curve saturate faster.

    :param torch.Tensor x: input tensor of linear values.
    :param float alpha: exposure-like parameter :math:`\alpha > 0`. Default: ``1.0``.
    :return: (:class:`torch.Tensor`) tone mapped values.
    """
    xc = torch.clamp(x, min=0.0) * alpha
    return xc / (1.0 + xc)


def reinhard_tone_inverse(y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    r"""
    Inverse Reinhard tone mapping :math:`x = \frac{y}{\alpha (1 - y)}`.

    This is the exact inverse of :func:`deepinv.physics.functional.reinhard_tone`
    on :math:`[0, \infty)`. Values of :math:`y \geq 1` (which are not in the
    range of the forward curve) are clamped to :math:`1 - \varepsilon`.

    :param torch.Tensor y: input tensor of tone mapped values.
    :param float alpha: exposure-like parameter :math:`\alpha > 0`. Default: ``1.0``.
    :return: (:class:`torch.Tensor`) linear values.
    """
    yc = torch.clamp(y, min=0.0, max=1.0 - 1e-6)
    return yc / (alpha * (1.0 - yc))
