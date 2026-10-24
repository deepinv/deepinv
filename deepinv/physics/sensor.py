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


class Saturation(SensorModel):
    r"""
    Saturation sensor model.

    Clips the measurements to a given dynamic range

    .. math::

        \eta(z) = \min\left(z_{\max}, \max\left(z_{\min}, z\right)\right).

    All values outside the dynamic range are mapped to its boundary, which
    makes the operator many-to-one: saturated measurements carry no
    information about the input signal beyond the range boundary. This is the
    standard model of display or storage clipping, e.g. of overexposed
    highlights in low dynamic range images.

    The operator is differentiable almost everywhere, with zero gradient
    outside the dynamic range. When it is included in the forward operator of
    a reconstruction (see :class:`deepinv.physics.SensorModel`), saturated
    measurements contribute no gradient to the data-fidelity term. This does not mean that the corresponding pixels
    are reconstructed from the prior alone: if the forward operator mixes
    neighbouring pixels (e.g. a blur), the underlying signal is still
    constrained by the neighbouring unsaturated measurements.

    :param tuple dynamic_range: dynamic range :math:`(z_{\min}, z_{\max})`.
        Default: ``(0.0, 1.0)``.

    :Examples:

        >>> import torch
        >>> from deepinv.physics import Saturation
        >>> sensor = Saturation()
        >>> sensor(torch.tensor([-0.5, 0.2, 3.0]))
        tensor([0.0000, 0.2000, 1.0000])
    """

    def __init__(self, dynamic_range: tuple = (0.0, 1.0)):
        super().__init__()
        self.dynamic_range = dynamic_range

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

    Quantizes the measurements to a uniform grid of :math:`L` values spanning
    the dynamic range :math:`[z_{\min}, z_{\max}]`

    .. math::

        \eta(z) = z_{\min} + \Delta \, \mathrm{round}\left(\frac{\mathrm{clip}(z) - z_{\min}}{\Delta}\right),
        \quad \Delta = \frac{z_{\max} - z_{\min}}{L - 1},

    where :math:`L` is the number of quantization levels (e.g. :math:`L=256`
    for 8-bit images) and :math:`\mathrm{clip}` clips to the dynamic range. If
    ``clip=False``, values outside the dynamic range are quantized on the same
    grid without clipping.

    Quantization is not differentiable (the gradient of rounding is zero
    almost everywhere). This operator therefore implements the straight-through
    estimator :footcite:p:`bengio2013estimating`: in the backward pass, the
    gradient is passed through unchanged for in-range values (and set to zero
    for clipped values, mirroring :class:`deepinv.physics.Saturation`).

    .. warning::

        If the quantizer is included in the forward operator of a
        reconstruction, the data-fidelity term is piecewise-constant: it is
        flat on the measurement-consistent set, so gradient-based
        reconstruction cannot explore it. For finely quantized measurements
        (e.g. 8-bit), it is usually better to leave the quantizer out of the
        forward operator and treat the quantization error as noise.

    Combined with :class:`deepinv.physics.ToneMapping`, this models the full
    low dynamic range measurement chain
    :math:`y = \mathrm{round}\left(\mathrm{clip}(\operatorname{TM}(x))\right)`.

    :param int levels: number of quantization levels :math:`L \geq 2`.
        Default: ``256``.
    :param tuple dynamic_range: dynamic range :math:`(z_{\min}, z_{\max})`.
        Default: ``(0.0, 1.0)``.
    :param bool clip: if ``True`` (default), clip the input to the dynamic
        range before quantizing.

    :Examples:

        >>> import torch
        >>> from deepinv.physics import Quantization
        >>> sensor = Quantization(levels=256)
        >>> sensor(torch.tensor([0.0, 0.5, 1.0, 2.0]))
        tensor([0.0000, 0.5020, 1.0000, 1.0000])
    """

    def __init__(
        self,
        levels: int = 256,
        dynamic_range: tuple = (0.0, 1.0),
        clip: bool = True,
    ):
        super().__init__()
        if isinstance(levels, bool) or not isinstance(levels, numbers.Integral):
            raise ValueError(f"levels must be an integer >= 2, got {levels}.")
        if levels < 2:
            raise ValueError(f"levels must be an integer >= 2, got {levels}.")
        self.levels = int(levels)
        self.dynamic_range = dynamic_range
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Quantizes the input to the uniform grid.

        :param torch.Tensor x: input tensor.
        :return: (:class:`torch.Tensor`) quantized tensor.
        """
        z_min, z_max = self.dynamic_range
        return _ClippedQuantization.apply(x, self.levels, z_min, z_max, self.clip)
