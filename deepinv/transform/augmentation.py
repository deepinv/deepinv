from __future__ import annotations

from typing import Union, Iterable

import torch
from torch import Tensor

from deepinv.transform.base import Transform, TransformParam
from deepinv.physics.mri import MRIMixin
from deepinv.physics.noise import GaussianNoise, NoiseModel


class RandomNoise(Transform):
    """Random noise transform.

    For now, only Gaussian noise is supported. Override this class and replace the `sigma` parameter for other noise models.

    This transform is reproducible: for given param dict `noise_model`, the transform is deterministic.

    Note the inverse transform is not well-defined for this transform.

    :param str noise_type: noise distribution, currently only supports Gaussian noise.
    :param int, tuple[int, int] sigma: noise parameter or range to pick randomly.
    """

    def __init__(
        self,
        *args,
        noise_type: str = "gaussian",
        sigma: Union[int, tuple[int, int]] = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        if noise_type == "gaussian":
            self.noise_class = GaussianNoise
        else:
            raise ValueError(f"Noise type {noise_type} not supported.")

    def _get_params(self, *args) -> dict:
        if isinstance(sr := self.sigma, tuple):
            sigma = (
                torch.rand(self.n_trans, generator=self.rng) * (sr[1] - sr[0])
            ) + sr[0]
        else:
            sigma = [self.sigma] * self.n_trans
        # TODO reproducible, different rng when self.n_trans > 1
        return {
            "noise_model": [
                self.noise_class(sigma=s, rng=self.rng if i == 0 else None)
                for i, s in enumerate(sigma)
            ]
        }

    def _transform(
        self, y: Tensor, noise_model: Iterable[NoiseModel] = [], **kwargs
    ) -> Tensor:
        mask = (y != 0).int()
        return torch.cat([n(y) * mask for n in noise_model])

    def inverse(self, *args, **kwargs):
        raise ValueError("Noise transform is not invertible.")


class RandomPhaseError(Transform):
    r"""Random phase error transform.

    This transform is specific to MRI problems, and adds a phase error to k-space using:

    :math:`Ty=\exp(-i\phi_k)y` where :math:`\phi_k=\pi\alpha s_e` if :math:`k` is an even index,
    or :math:`\phi_k=\pi\alpha s_o` if odd, and where :math:`\alpha` is a scale parameter,
    and :math:`s_o,s_e\sim U(-1,1)`.

    This transform is reproducible: for given param dict `se, so`, the transform is deterministic.

    :param int, tuple[int, int] scale: scale parameters :math:`s_e` and :math:`s_o` or range to pick randomly.
    """

    def __init__(self, *args, scale: Union[int, tuple[int, int]] = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.flatten_video_input = False

    def _get_params(self, *args) -> dict:
        if isinstance(s := self.scale, tuple):
            scale = (
                torch.rand((1, self.n_trans), generator=self.rng) * (s[1] - s[0])
            ) + s[0]
        else:
            scale = self.scale

        se, so = (
            2
            * torch.pi
            * scale
            * torch.rand((2, self.n_trans), generator=self.rng, device=self.rng.device)
            - torch.pi * scale
        )
        return {"se": se, "so": so}

    def _transform(
        self,
        y,
        se: Union[torch.Tensor, Iterable, TransformParam] = [],
        so: Union[torch.Tensor, Iterable, TransformParam] = [],
        **kwargs,
    ) -> Tensor:
        out = []
        for _se, _so in zip(se, so, strict=True):
            shift = MRIMixin.to_torch_complex(torch.zeros_like(y))
            shift[..., 0::2] = torch.exp(-1j * _se)  # assume readouts in w
            shift[..., 1::2] = torch.exp(-1j * _so)
            out += [y * MRIMixin.from_torch_complex(shift)]
        return torch.cat(out)
