from __future__ import annotations
import torch
from torch import Tensor
from .base import Denoiser
from .utils import array2tensor, tensor2array


class BM3D(Denoiser):
    r"""
    BM3D denoiser.

    The BM3D denoiser was introduced by :footcite:t:`dabov2007image`.


    .. note::

        Unlike other denoisers from the library, this denoiser is applied sequentially to each noisy image in the batch
        (no parallelization). Furthermore, it does not support backpropagation.

    .. warning::

        This module wraps the BM3D denoiser from the `BM3D python package <https://pypi.org/project/bm3d/>`_.
        It can be installed with ``pip install bm3d``.

    """

    def forward(self, x: Tensor, sigma: float | Tensor, **kwargs) -> Tensor:
        try:
            import bm3d
        except ImportError:  # pragma: no cover
            raise ImportError(
                "bm3d package not found. Please install it with `pip install bm3d`."
            )

        out = torch.empty_like(x)

        sigma = self._handle_sigma(sigma, batch_size=x.size(0))

        for i in range(x.shape[0]):
            out[i, :, :, :] = array2tensor(
                bm3d.bm3d(tensor2array(x[i, :, :, :]), sigma[i].item())
            )
        return out
