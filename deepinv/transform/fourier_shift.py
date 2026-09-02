from __future__ import annotations

from collections.abc import Iterable

import torch

from deepinv.transform.base import Transform, TransformParam


class FourierShift(Transform):
    r"""Continuous circular translations using the Fourier shift theorem.

    Unlike :class:`deepinv.transform.Shift`, which uses integer pixel rolls,
    this transform accepts floating-point translations. Under the periodic,
    band-limited discrete-image model, its inverse is its exact adjoint for
    complex tensors and two-channel real/imaginary tensors.

    :param float shift_max: maximum random shift as a fraction of image height
        and width, defaults to 1.0.
    :param int n_trans: number of transformed versions per input image.
    :param torch.Generator rng: random number generator.
    """

    def __init__(self, *args, shift_max: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if shift_max < 0:
            raise ValueError("shift_max must be nonnegative.")
        self.shift_max = shift_max

    def _get_params(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = x.shape[-2:]

        def sample(maximum: float) -> torch.Tensor:
            return maximum * (
                2
                * torch.rand(
                    self.n_trans,
                    generator=self.rng,
                    device=self.rng.device,
                )
                - 1
            )

        return {
            "x_shift": sample(self.shift_max * width),
            "y_shift": sample(self.shift_max * height),
        }

    @staticmethod
    def _shift(
        x: torch.Tensor,
        x_shift: torch.Tensor | float,
        y_shift: torch.Tensor | float,
    ) -> torch.Tensor:
        two_channel_complex = not x.is_complex() and x.shape[1] == 2
        if two_channel_complex:
            x = torch.view_as_complex(x.moveaxis(1, -1).contiguous())
        height, width = x.shape[-2:]
        real_dtype = x.real.dtype if x.is_complex() else x.dtype
        x_shift = torch.as_tensor(x_shift, device=x.device, dtype=real_dtype)
        y_shift = torch.as_tensor(y_shift, device=x.device, dtype=real_dtype)
        fy = torch.fft.fftfreq(height, device=x.device, dtype=real_dtype)
        fx = torch.fft.fftfreq(width, device=x.device, dtype=real_dtype)
        phase = torch.exp(
            -2j * torch.pi * (y_shift * fy[:, None] + x_shift * fx[None, :])
        )
        shifted = torch.fft.ifftn(
            torch.fft.fftn(x, dim=(-2, -1)) * phase,
            dim=(-2, -1),
        )
        if two_channel_complex:
            return torch.view_as_real(shifted).moveaxis(-1, 1)
        return shifted if x.is_complex() else shifted.real

    def _transform(
        self,
        x: torch.Tensor,
        x_shift: torch.Tensor | Iterable | TransformParam = tuple(),
        y_shift: torch.Tensor | Iterable | TransformParam = tuple(),
        **kwargs,
    ) -> torch.Tensor:
        return torch.cat(
            [self._shift(x, sx, sy) for sx, sy in zip(x_shift, y_shift, strict=True)],
            dim=0,
        )
