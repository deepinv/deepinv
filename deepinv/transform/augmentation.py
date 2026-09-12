from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from deepinv.transform.base import Transform, TransformParam
from deepinv.utils.mixins import MRIMixin
from deepinv.physics.noise import GaussianNoise


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
        sigma: int | tuple[int, int] = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        if noise_type == "gaussian":
            self.noise_class = GaussianNoise
        else:
            raise ValueError(f"Noise type {noise_type} not supported.")

    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        """Perform random transformation on image.

        Calls ``get_params`` to generate random params for image, then ``transform`` to deterministically transform.

        For purely deterministic transformation, pass in custom params and ``get_params`` will be ignored.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :return torch.Tensor: randomly transformed images concatenated along the first dimension
        """
        if params:
            raise ValueError(
                f"{self.__class__.__name__} is not a parametrized transform, cannot pass in params."
            )

        if self._check_x_5D(x) and self.flatten_video_input:
            shape = x.shape[1:]
            x = self.flatten_C(x)
            out_reshape = (-1, *shape)
        else:
            out_reshape = None

        if isinstance(sr := self.sigma, tuple):
            sigma = (
                torch.rand(self.n_trans, generator=self.rng) * (sr[1] - sr[0])
            ) + sr[0]
        else:
            sigma = [self.sigma] * self.n_trans

        # TODO reproducible, different rng when self.n_trans > 1
        noise_model = [
            self.noise_class(sigma=s, rng=self.rng if i == 0 else None)
            for i, s in enumerate(sigma)
        ]

        mask = (x != 0).int()
        out = torch.cat([n(x) * mask for n in noise_model])

        if out_reshape is not None:
            out = out.reshape(out_reshape)

        return out

    def inverse(self, *args, **kwargs):
        raise ValueError(
            f"{self.__class__.__name__} is not a parametrized transform, cannot invert."
        )

    def get_params(self, x: torch.Tensor) -> dict:
        raise ValueError(
            f"{self.__class__.__name__} is not a parametrized transform, cannot get params."
        )


class RandomPhaseError(Transform):
    r"""Random phase error transform.

    This transform is specific to MRI problems, and adds a phase error to k-space using:

    :math:`Ty=\exp(-i\phi_k)y` where :math:`\phi_k=\pi\alpha s_e` if :math:`k` is an even index,
    or :math:`\phi_k=\pi\alpha s_o` if odd, and where :math:`\alpha` is a scale parameter,
    and :math:`s_o,s_e\sim U(-1,1)`.

    This transform is reproducible: for given param dict `se, so`, the transform is deterministic.

    :param int, tuple[int, int] scale: scale parameters :math:`s_e` and :math:`s_o` or range to pick randomly.
    """

    def __init__(self, *args, scale: int | tuple[int, int] = 0.2, **kwargs):
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
        se: torch.Tensor | Iterable | TransformParam = tuple(),
        so: torch.Tensor | Iterable | TransformParam = tuple(),
        **kwargs,
    ) -> Tensor:
        out = []
        for _se, _so in zip(se, so, strict=True):
            shift = MRIMixin.to_torch_complex(torch.zeros_like(y))
            shift[..., 0::2] = torch.exp(-1j * _se)  # assume readouts in w
            shift[..., 1::2] = torch.exp(-1j * _so)
            out += [y * MRIMixin.from_torch_complex(shift)]
        return torch.cat(out)
