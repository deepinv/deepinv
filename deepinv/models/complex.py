from __future__ import annotations

from .base import Denoiser
from .wrapper import ComplexDenoiserWrapper


def to_complex_denoiser(denoiser: Denoiser, mode="real_imag") -> ComplexDenoiserWrapper:
    r"""
    Converts a denoiser with real inputs into the one with complex inputs.

    Converts a denoiser with real inputs into one that accepts complex-valued inputs by applying the denoiser separately on the real and imaginary parts, or in the absolute value and phase parts.

    :param torch.nn.Module denoiser: a denoiser which takes in real-valued inputs.
    :param str mode: the mode by which the complex inputs are processed. Can be either `'real_imag'` or `'abs_angle'`.
    :return: (torch.nn.Module) the denoiser which takes in complex-valued inputs.
    """
    return ComplexDenoiserWrapper(denoiser, mode)
