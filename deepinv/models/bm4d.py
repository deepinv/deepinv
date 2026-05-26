from __future__ import annotations
import torch
from torch import Tensor
from .base import Denoiser


class BM4D(Denoiser):
    r"""
    BM4D denoiser.

    The BM4D denoiser is a generalization of the BM3D denoiser (introduced by :footcite:t:`dabov2007image`) to volumetric images.


    .. note::

        Unlike other denoisers from the library, this denoiser is applied sequentially to each noisy image in the batch
        (no parallelization). Furthermore, it does not support backpropagation.

    .. warning::

        This module wraps the BM4D denoiser from the `BM4D python package <https://pypi.org/project/bm4d/>`_.
        It can be installed with ``pip install bm4d``.

    """

    def forward(self, x: Tensor, sigma: float | Tensor, **kwargs) -> Tensor:
        try:
            import bm4d
        except ImportError:  # pragma: no cover
            raise ImportError(
                "bm4d package not found. Please install it with `pip install bm4d`."
            )

        out = torch.empty_like(x)

        sigma = self._handle_sigma(sigma, batch_size=x.size(0))

        for i in range(x.shape[0]):
            out[i, :, :, :, :] = torch.from_numpy(
                bm4d.bm4d_multichannel(
                    x[i, :, :, :, :].cpu().detach().numpy(),
                    sigma[i].item(),
                )
            )
        return out
