from __future__ import annotations
import torch
from .base import Denoiser


def to_complex_denoiser(denoiser: Denoiser, mode="real_imag") -> ComplexDenoiser:
    r"""
    Converts a denoiser with real inputs into the one with complex inputs.

    Converts a denoiser with real inputs into one that accepts complex-valued inputs by applying the denoiser separately on the real and imaginary parts, or in the absolute value and phase parts.

    :param torch.nn.Module denoiser: a denoiser which takes in real-valued inputs.
    :param str mode: the mode by which the complex inputs are processed. Can be either `'real_imag'` or `'abs_angle'`.
    :return: (torch.nn.Module) the denoiser which takes in complex-valued inputs.
    """
    return ComplexDenoiserWrapper(denoiser, mode)


class ComplexDenoiserWrapper(Denoiser):
    r"""
    A wrapper class to convert a real-valued denoiser into a complex-valued denoiser.
    It processes complex inputs by splitting them into real and imaginary parts or into absolute value and phase parts, applying the real-valued denoiser separately, and then recombining the results. 
    
    :param deepinv.models.Denoiser denoiser: a denoiser which takes in real-valued inputs.
    :param str mode: the mode by which the complex inputs are processed. Can be either `'real_imag'` or `'abs_angle'`. If `'real_imag'`, the denoiser is applied separately on the real and imaginary parts. If `'abs_angle'`, the denoiser is applied separately on the absolute value and phase parts. Default is `'real_imag'`.
    
    
    
    """
    def __init__(
        self, denoiser: Denoiser, mode: str = "real_imag", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.denoiser = denoiser
        
        if mode.lower() not in ["real_imag", "abs_angle"]:
            raise ValueError(f"'mode' must be 'real_imag' or 'abs_angle'. Got {mode} instead.")

    def forward(self, x: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        r"""
        Applies the complex-valued denoiser. If a real tensor is provided, it is treated as a complex tensor with zero imaginary part.
        
        :param torch.Tensor x: complex-valued input images.
        :param float or torch.Tensor sigma: noise level.

        :returns: (:class:`torch.Tensor`) Denoised images, with the same shape as the input and will always be in complex dtype. 
        """
        if self.mode == "real_imag":
            x_real = x.real

            if torch.is_complex(x):
                noisy_batch = torch.cat((x_real, x.imag), 0)
                denoised_batch = self.denoiser(noisy_batch, sigma)
                return (
                denoised_batch[: x_real.shape[0], ...]
                + 1j * denoised_batch[x_real.shape[0] :, ...]
            )
            else:
                return self.denoiser(x_real) + 0j
                        
        else:  # abs_angle
            x_mag = torch.abs(x)
            x_phase = torch.angle(x)
            noisy_batch = torch.cat((x_mag, x_phase), 0)
            denoised_batch = self.denoiser(noisy_batch, sigma)
            return denoised_batch[: x_mag.shape[0], ...] * torch.exp(
                1j * denoised_batch[x_mag.shape[0] :, ...]
            )
