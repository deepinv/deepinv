from __future__ import annotations

import numbers
from typing import Callable

import torch
import torch.nn as nn


class SensorModel(nn.Module):
    r"""
    Base class for sensor models.

    Sensor models are deterministic non-linearities :math:`\eta` applied to
    the measurements, after the noise, such as

    .. math::

        y = \eta\left(N(A(x))\right).

    They can be passed to any physics via the ``sensor_model`` argument (see
    :class:`deepinv.physics.Physics`), and are then applied when simulating
    measurements with ``physics(x)``.

    .. note::

        Reconstruction algorithms and data-fidelity terms only use the forward
        operator :math:`A`, and ignore the sensor model. To account for a
        sensor non-linearity during reconstruction, include it in the forward
        operator, e.g. in the curve of a :class:`deepinv.physics.ToneMapping`
        operator that is composed with the other operators.

    :param Callable sensor_model: sensor non-linearity function :math:`\eta(z)`.
    """

    def __init__(self, sensor_model: Callable = None):
        super().__init__()
        if sensor_model is None:
            sensor_model = lambda x: x
        self.sensor_model = sensor_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Applies the sensor non-linearity.

        :param torch.Tensor x: input tensor.
        :return: (:class:`torch.Tensor`) output tensor.
        """
        return self.sensor_model(x)


def _check_dynamic_range(dynamic_range) -> tuple:
    if len(dynamic_range) != 2:
        raise ValueError(
            f"dynamic_range must be a tuple (z_min, z_max), got {dynamic_range!r}."
        )
    z_min, z_max = dynamic_range
    if not z_min < z_max:
        raise ValueError(
            f"dynamic_range must satisfy z_min < z_max, got {dynamic_range!r}."
        )
    return tuple(dynamic_range)


class Saturation(SensorModel):
    r"""
    Saturation sensor model: clips the measurements to a dynamic range.

    .. math::

        \eta(z) = \min\left(z_{\max}, \max\left(z_{\min}, z\right)\right).

    Values outside the range map to its boundary, making the operator many-to-one: saturated measurements carry no information beyond the boundary (e.g. overexposed highlights in LDR images). Differentiable almost everywhere, with zero gradient outside the range, so saturated measurements contribute no gradient to the data-fidelity term; if the forward operator mixes pixels (e.g. a blur), the signal is still constrained by neighbouring unsaturated measurements.

    :param tuple dynamic_range: dynamic range :math:`(z_{\min}, z_{\max})`, with :math:`z_{\min} < z_{\max}`. Default: ``(0.0, 1.0)``.
    """

    def __init__(self, dynamic_range: tuple = (0.0, 1.0)):
        super().__init__()
        self.dynamic_range = _check_dynamic_range(dynamic_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Clips the input to the dynamic range.

        :param torch.Tensor x: input tensor.
        :return: (:class:`torch.Tensor`) clipped tensor.
        """
        z_min, z_max = self.dynamic_range
        return torch.clamp(x, z_min, z_max)


class _ClippedQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, levels, z_min, z_max, clip):
        if clip:
            ctx.save_for_backward((x >= z_min) & (x <= z_max))
            x = torch.clamp(x, z_min, z_max)
        else:
            ctx.save_for_backward(torch.ones_like(x, dtype=torch.bool))
        scale = (levels - 1) / (z_max - z_min)
        return z_min + torch.round((x - z_min) * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask.to(grad_output.dtype), None, None, None, None


class Quantization(SensorModel):
    r"""
    Quantization sensor model with straight-through gradient estimator.

    Quantizes the measurements to a uniform grid spanning the dynamic range :math:`[z_{\min}, z_{\max}]`:

    .. math::

        \eta(z) = z_{\min} + \Delta \, \mathrm{round}\left(\frac{\mathrm{clip}(z) - z_{\min}}{\Delta}\right),
        \quad \Delta = \frac{z_{\max} - z_{\min}}{L - b - 1},

    where :math:`L` is the number of ADC levels of the sensor and :math:`b` its black level, so the usable dynamic range spans :math:`L - b` levels. Set :math:`L` directly with ``levels`` or from the bit depth with ``bits`` (:math:`L = 2^{bits}`), e.g. ``bits=10``, ``12``, ``14`` or ``16`` for typical cameras. If ``clip=False``, values outside the dynamic range are quantized on the same grid without clipping.

    .. note::

        Rounding has zero gradient almost everywhere. This class implements the straight-through estimator :footcite:p:`bengio2013estimating`: the gradient is passed through unchanged for in-range values and set to zero for clipped values, mirroring :class:`deepinv.physics.Saturation`.

    .. warning::

        If the quantizer is part of the forward operator of a reconstruction, the data-fidelity term is piecewise-constant, so gradient-based reconstruction cannot explore it. For finely quantized measurements (e.g. 8-bit), leave the quantizer out of the forward operator and model the quantization error as noise.

    :param int, None levels: number of ADC levels :math:`L` spanning the dynamic range. The number of quantization levels is ``levels - black_level`` and must be >= 2. Mutually exclusive with ``bits``. If neither ``levels`` nor ``bits`` is passed, defaults to ``256`` (8-bit).
    :param int, None bits: bit depth of the sensor, sets ``levels = 2**bits`` (``bits=8`` is equivalent to ``levels=256``). Mutually exclusive with ``levels``.
    :param float, list, tuple, torch.Tensor black_level: black level :math:`b \geq 0` in raw ADC units, i.e. the sensor output at zero intensity (black level error). A list, tuple or 1D tensor sets one black level per channel, and its length must match the channel dimension (dim 1) of the input. Black levels that vary over the pixels of a mosaiced raw image (e.g. per Bayer position) are not supported. Reduces the usable dynamic range from :math:`L` to :math:`L - b` levels, i.e. :math:`2^{bits} - b` when ``bits`` is used.
    :param tuple dynamic_range: dynamic range :math:`(z_{\min}, z_{\max})`, with :math:`z_{\min} < z_{\max}`. Default: ``(0.0, 1.0)``.
    :param bool clip: if ``True``, clip the input to the dynamic range before quantizing. Default: ``True``.
    """

    def __init__(
        self,
        levels: int | None = None,
        bits: int | None = None,
        dynamic_range: tuple = (0.0, 1.0),
        clip: bool = True,
        black_level: float | list | tuple | torch.Tensor = 0,
    ):
        super().__init__()
        if levels is not None and bits is not None:
            raise ValueError(
                f"levels and bits are mutually exclusive, got levels={levels} and bits={bits}."
            )
        if bits is not None:
            if isinstance(bits, bool) or not isinstance(bits, numbers.Integral):
                raise TypeError(
                    f"bits must be an int, but got type '{type(bits).__name__}'."
                )
            bits = int(bits)
            if bits < 1:
                raise ValueError(f"bits must be an int >= 1, got {bits}.")
            levels = 2**bits
            self.bits = bits
        else:
            self.bits = None
        if levels is None:
            levels = 256
        if isinstance(levels, bool) or not isinstance(levels, numbers.Integral):
            raise TypeError(
                f"levels must be an int, but got type '{type(levels).__name__}'."
            )
        levels = int(levels)
        if levels < 2:
            raise ValueError(f"levels must be an int >= 2, got {levels}.")
        if isinstance(black_level, (list, tuple)):
            if any(
                isinstance(b, bool) or not isinstance(b, numbers.Real)
                for b in black_level
            ):
                raise TypeError(
                    f"black_level must contain only real numbers, but got {black_level!r}."
                )
            black_level = torch.as_tensor(black_level)
        if torch.is_tensor(black_level):
            if black_level.dtype == torch.bool or black_level.is_complex():
                raise TypeError(
                    f"black_level tensor must have a real numeric dtype, but got '{black_level.dtype}'."
                )
            negative = bool((black_level < 0).any())
            too_few = bool((levels - black_level < 2).any())
        else:
            if isinstance(black_level, bool) or not isinstance(
                black_level, numbers.Real
            ):
                raise TypeError(
                    f"black_level must be an int, float, list, tuple or torch.Tensor, but got type '{type(black_level).__name__}'."
                )
            negative = black_level < 0
            too_few = levels - black_level < 2
        if negative:
            raise ValueError(f"black_level must be >= 0, got {black_level}.")
        if too_few:
            raise ValueError(
                f"levels - black_level must be >= 2, got {levels} - {black_level}."
            )
        if torch.is_tensor(black_level):
            if black_level.ndim > 1:
                raise ValueError(
                    f"black_level tensor must be 0D or 1D, got shape {tuple(black_level.shape)}."
                )
            # buffers, so that they follow the module across devices and are saved
            self.register_buffer("black_level", black_level)
            self.register_buffer("levels", levels - black_level)
        else:
            self.black_level = black_level
            self.levels = levels - black_level
        self.dynamic_range = _check_dynamic_range(dynamic_range)
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Quantizes the input to the uniform grid.

        :param torch.Tensor x: input tensor.
        :return: (:class:`torch.Tensor`) quantized tensor.
        """
        z_min, z_max = self.dynamic_range
        levels = self.levels
        if torch.is_tensor(levels):
            levels = levels.to(x.device)
            if levels.ndim == 1:
                # one black level per channel: broadcast over the channel dim
                if x.ndim < 2 or x.shape[1] != levels.numel():
                    raise ValueError(
                        f"Got {levels.numel()} per-channel black levels, but the "
                        f"input has shape {tuple(x.shape)} (channels in dim 1)."
                    )
                levels = levels.view(1, -1, *[1] * (x.ndim - 2))
        return _ClippedQuantization.apply(x, levels, z_min, z_max, self.clip)
