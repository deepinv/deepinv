import torch


def to_complex_denoiser(denoiser, mode="real_imag"):
    r"""
    Given a denoiser, returns a corresponding modified denoiser which can process complex numbers.

    :param torch.nn.Module denoiser: a denoiser which takes in real-valued inputs.
    :param str mode: the mode by which the complex inputs are processed. Can be either 'real_imag' or 'abs_angle'.
    :return: (torch.nn.Module) the denoiser which takes in complex-valued inputs.
    """

    class complex_denoiser(torch.nn.Module):
        def __init__(self, denoiser: torch.nn.Module, mode, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mode = mode
            self.denoiser = denoiser

        def forward(self, x, sigma):
            if self.mode == "real_imag":
                x_real = x.real
                x_imag = x.imag
                x_real = self.denoiser.forward(x_real, sigma)
                x_imag = self.denoiser.forward(x_imag, sigma)
                return x_real + 1j * x_imag
            elif self.mode == "abs_angle":
                x_mag = torch.abs(x)
                x_phase = torch.angle(x)
                x_mag = self.denoiser.forward(x_mag, sigma)
                x_phase = self.denoiser.forward(x_phase, sigma)
                return x_mag * torch.exp(1j * x_phase)
            else:
                raise ValueError("style must be 'real_imag' or 'abs_angle'.")

    return complex_denoiser(denoiser, mode)
