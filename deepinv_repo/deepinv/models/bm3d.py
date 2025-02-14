import numpy as np
import torch
from .base import Denoiser

# Compat for optional dependency on BM3D
try:
    import bm3d
except:
    bm3d = ImportError("The bm3d package is not installed.")


class BM3D(Denoiser):
    r"""
    BM3D denoiser.

    The BM3D denoiser was introduced in "Image denoising by sparse 3D transform-domain collaborative filtering", by
    Dabov et al., IEEE Transactions on Image Processing (2007).


    .. note::

        Unlike other denoisers from the library, this denoiser is applied sequentially to each noisy image in the batch
        (no parallelization). Furthermore, it does not support backpropagation.

    .. warning::

        This module wraps the BM3D denoiser from the `BM3D python package <https://pypi.org/project/bm3d/>`_.
        It can be installed with ``pip install bm3d``.

    """

    def __init__(self):
        super().__init__()
        if isinstance(bm3d, ImportError):
            raise ImportError(
                "BM3D denoiser not available. Please install the bm3d package with `pip install bm3d`."
            ) from bm3d

    def forward(self, x, sigma, **kwargs):
        r"""
        Run the denoiser on image with noise level :math:`\sigma`.

        :param torch.Tensor x: noisy image
        :param float sigma: noise level
        """

        out = torch.zeros_like(x)

        for i in range(x.shape[0]):
            out[i, :, :, :] = array2tensor(
                bm3d.bm3d(tensor2array(x[i, :, :, :]), sigma)
            )
        return out


def tensor2array(img):
    img = img.cpu().detach().numpy()
    if img.shape[0] == 3:  # Color case: cast to BM3D format (W,H,C)
        img = np.transpose(img, (1, 2, 0))
    else:  # Grayscale case: cast to BM3D format (W,H)
        img = img[0]
    return img


def array2tensor(img):
    if len(img.shape) == 3:  # Color case: back to (C,W,H)
        out = torch.from_numpy(img).permute(2, 0, 1)
    else:  # Grayscale case: back to (1,W,H)
        out = torch.from_numpy(img).unsqueeze(0)
    return out
