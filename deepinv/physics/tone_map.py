from __future__ import annotations

from typing import Callable, ClassVar

import torch

from deepinv.physics.forward import Physics
from deepinv.physics.functional.tone_map import (
    gamma_tone,
    gamma_tone_inverse,
    log_tone,
    log_tone_inverse,
    reinhard_tone,
    reinhard_tone_inverse,
    srgb_eotf,
    srgb_oetf,
)


class ToneMapping(Physics):
    r"""
    Tone mapping operator.

    Tone mapping operators map high dynamic range (HDR) linear images
    :math:`x \in \mathbb{R}^n` to low dynamic range (LDR) images
    :math:`y \in [0, 1]^n`. Recovering the HDR image from an LDR measurement is
    an ill-posed inverse problem known as inverse tone mapping. The forward
    operator is a pointwise, monotone increasing and differentiable curve

    .. math::

        y = \operatorname{TM}(x).

    Since the curve is pointwise and monotone, the pseudo-inverse
    :meth:`A_dagger` is available in closed form as the exact inverse curve,
    and the VJP :meth:`deepinv.physics.Physics.A_vjp` is computed exactly via
    automatic differentiation.

    Predefined curves can be selected by name:

    - ``"srgb"``: sRGB OETF of the sRGB standard :footcite:t:`iec1999srgb`
      (:func:`deepinv.physics.functional.srgb_oetf`), e.g. for modelling the
      linear-to-non-linear conversion of a camera ISP,
    - ``"gamma"``: power-law gamma encoding
      (:func:`deepinv.physics.functional.gamma_tone`),
    - ``"log"``: logarithmic tone mapping
      (:func:`deepinv.physics.functional.log_tone`),
    - ``"reinhard"``: Reinhard tone mapping
      (:func:`deepinv.physics.functional.reinhard_tone`).

    A custom curve can be passed as a callable, optionally together with its
    closed-form inverse. If no inverse is provided, :meth:`A_dagger` falls back
    to gradient descent, which requires an initial guess ``x_init``.

    This operator can also be used as a :math:`\eta` in the ``sensor_model`` of
    another physics (e.g. for modelling the sRGB encoding applied after optical
    degradations and noise), or composed with other physics via
    :func:`deepinv.physics.compose`.

    :param str, Callable curve: name of a predefined curve (``"srgb"``,
        ``"gamma"``, ``"log"`` or ``"reinhard"``) or a custom differentiable
        callable. Default: ``"srgb"``.
    :param dict curve_params: optional keyword arguments passed to the curve
        and its inverse (e.g. ``{"gamma": 2.4}`` for ``"gamma"``,
        ``{"L": 100.0}`` for ``"log"`` or ``{"alpha": 1.0}`` for
        ``"reinhard"``). For a custom curve, they are passed to the callable
        and its inverse.
    :param Callable inverse: optional closed-form inverse of a custom curve.
        Ignored for predefined curves.

    :Examples:

        Tone map an HDR image and recover it exactly via the inverse curve:

        >>> import torch
        >>> from deepinv.physics import ToneMapping
        >>> physics = ToneMapping(curve="reinhard")
        >>> x = torch.rand(1, 3, 8, 8) * 100.0  # HDR image
        >>> y = physics(x)
        >>> torch.max(y) <= 1.0
        tensor(True)
        >>> x_hat = physics.A_dagger(y)
        >>> torch.allclose(x_hat, x)
        True

        Use the sRGB OETF as the sensor non-linearity of a blur physics, i.e.
        :math:`y = \operatorname{OETF}(A(x))`:

        >>> import torch
        >>> from deepinv.physics import Blur
        >>> from deepinv.physics.functional import srgb_oetf
        >>> physics = Blur(filter=torch.ones(3, 3) / 9, sensor_model=srgb_oetf)

        Model the full LDR acquisition chain, with clipping and 8-bit
        quantization applied after the tone mapping curve:

        >>> import torch
        >>> from deepinv.physics import ToneMapping, Quantization
        >>> physics = ToneMapping(curve="srgb", sensor_model=Quantization(levels=256))
        >>> x = torch.rand(1, 3, 8, 8) * 10.0  # HDR image
        >>> y = physics(x)
        >>> torch.max(y) <= 1.0
        tensor(True)
    """

    _CURVES: ClassVar[dict] = {
        "srgb": (srgb_oetf, srgb_eotf),
        "gamma": (gamma_tone, gamma_tone_inverse),
        "log": (log_tone, log_tone_inverse),
        "reinhard": (reinhard_tone, reinhard_tone_inverse),
    }

    def __init__(
        self,
        curve: str | Callable = "srgb",
        curve_params: dict | None = None,
        inverse: Callable | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.curve_params = dict(curve_params) if curve_params is not None else {}

        if callable(curve):
            self.curve_name = None
            if inverse is not None and not callable(inverse):
                raise ValueError("inverse must be a callable.")
            self.curve_fn = curve
            self.inverse_fn = inverse
        elif isinstance(curve, str):
            if curve not in self._CURVES:
                raise ValueError(
                    f"Unknown curve {curve}. Choose from {list(self._CURVES)} or pass a callable."
                )
            self.curve_name = curve
            self.curve_fn, self.inverse_fn = self._CURVES[curve]
        else:
            raise ValueError(
                "curve must be a string or a callable, got " f"{type(curve).__name__}."
            )

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Applies the tone mapping curve.

        .. math::

            y = \operatorname{TM}(x)

        :param torch.Tensor x: HDR image.
        :return: (:class:`torch.Tensor`) LDR measurements.
        """
        return self.curve_fn(x, **self.curve_params)

    def A_dagger(self, y: torch.Tensor, x_init=None, **kwargs) -> torch.Tensor:
        r"""
        Applies the inverse tone mapping curve.

        For predefined curves (or a custom curve with a provided inverse), the
        inverse is exact and computed in closed form.

        :param torch.Tensor y: LDR measurements.
        :param None, torch.Tensor x_init: initial guess for the gradient descent
            fallback. Ignored when a closed-form inverse is available.
        :return: (:class:`torch.Tensor`) estimated HDR image.
        """
        if self.inverse_fn is not None:
            return self.inverse_fn(y, **self.curve_params)
        return super().A_dagger(y, x_init=x_init)
