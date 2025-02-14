import torch
from .base import Denoiser
from typing import Union


def to_complex_denoiser(denoiser, mode="real_imag"):
    r"""
    Converts a denoiser with real inputs into the one with complex inputs.

    Converts a denoiser with real inputs into one that accepts complex-valued inputs by applying the denoiser separately on the real and imaginary parts, or in the absolute value and phase parts.

    :param torch.nn.Module denoiser: a denoiser which takes in real-valued inputs.
    :param str mode: the mode by which the complex inputs are processed. Can be either `'real_imag'` or `'abs_angle'`.
    :return: (torch.nn.Module) the denoiser which takes in complex-valued inputs.
    """

    class complex_denoiser(Denoiser):
        def __init__(
            self, denoiser: Union[torch.nn.Module, Denoiser], mode: str, *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.mode = mode
            self.denoiser = denoiser

        def forward(self, x, sigma=None):
            if self.mode == "real_imag":
                x_real = x.real
                x_imag = x.imag
                noisy_batch = torch.cat((x_real, x_imag), 0)
                denoised_batch = self.denoiser(noisy_batch, sigma)
                return (
                    denoised_batch[: x_real.shape[0], ...]
                    + 1j * denoised_batch[x_real.shape[0] :, ...]
                )
            elif self.mode == "abs_angle":
                x_mag = torch.abs(x)
                x_phase = torch.angle(x)
                noisy_batch = torch.cat((x_mag, x_phase), 0)
                denoised_batch = self.denoiser(noisy_batch, sigma)
                return denoised_batch[: x_mag.shape[0], ...] * torch.exp(
                    1j * denoised_batch[x_mag.shape[0] :, ...]
                )
            else:
                raise ValueError("style must be 'real_imag' or 'abs_angle'.")

    return complex_denoiser(denoiser, mode)
