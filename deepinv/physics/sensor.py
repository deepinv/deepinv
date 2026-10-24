from __future__ import annotations

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
    :class:`deepinv.physics.Physics`). Unlike the forward operator ``A``, sensor
    models are not required to be differentiable: they only need to be
    differentiable if they are used inside gradient-based reconstruction, e.g.
    in a sensor-aware data-fidelity term
    :math:`\| \eta(A(x)) - y \|^2`.

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
    outside the dynamic range. Reconstruction algorithms can therefore
    differentiate through it; saturated pixels simply receive no gradient
    from the data-fidelity term and are reconstructed from the prior alone.

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
    def forward(ctx, x, levels, clip):
        if clip:
            ctx.save_for_backward((x >= 0.0) & (x <= 1.0))
            x = torch.clamp(x, 0.0, 1.0)
        else:
            ctx.save_for_backward(torch.ones_like(x, dtype=torch.bool))
        return torch.round(x * (levels - 1)) / (levels - 1)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask.to(grad_output.dtype), None, None


class Quantization(SensorModel):
    r"""
    Quantization sensor model with straight-through gradient estimator.

    Quantizes the measurements to a uniform grid of ``levels`` values

    .. math::

        \eta(z) = \frac{\mathrm{round}\left((L-1) \, \mathrm{clip}(z)\right)}{L-1},

    where :math:`L` is the number of quantization levels (e.g. :math:`L=256`
    for 8-bit images). If ``clip=False``, the input is quantized without
    clipping to :math:`[0, 1]`.

    Quantization is not differentiable (the gradient of rounding is zero
    almost everywhere). For use in gradient-based reconstruction, this operator
    implements the straight-through estimator :footcite:p:`bengio2013estimating`:
    in the backward pass, the gradient is passed through unchanged for
    in-range values (and set to zero for clipped values, mirroring
    :class:`deepinv.physics.Saturation`).

    Combined with :class:`deepinv.physics.ToneMapping`, this models the full
    low dynamic range measurement chain
    :math:`y = \mathrm{round}\left(\mathrm{clip}(\operatorname{TM}(x))\right)`.

    :param int levels: number of quantization levels :math:`L \geq 2`.
        Default: ``256``.
    :param bool clip: if ``True`` (default), clip the input to :math:`[0, 1]`
        before quantizing.

    :Examples:

        >>> import torch
        >>> from deepinv.physics import Quantization
        >>> sensor = Quantization(levels=256)
        >>> sensor(torch.tensor([0.0, 0.5, 1.0, 2.0]))
        tensor([0.0000, 0.5020, 1.0000, 1.0000])
    """

    def __init__(self, levels: int = 256, clip: bool = True):
        super().__init__()
        if not isinstance(levels, int) or levels < 2:
            raise ValueError(f"levels must be an integer >= 2, got {levels}.")
        self.levels = levels
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Quantizes the input to the uniform grid.

        :param torch.Tensor x: input tensor.
        :return: (:class:`torch.Tensor`) quantized tensor.
        """
        return _ClippedQuantization.apply(x, self.levels, self.clip)
