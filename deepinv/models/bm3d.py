import bm3d
import numpy as np
import torch
import torch.nn as nn


class BM3D(nn.Module):
    """
    BM3D denoiser.


    This module wraps the BM3D denoiser from the `BM3D python package <https://pypi.org/project/bm3d/>`_.
    The denoiser is applied sequentially to each noisy image in the batch.

    The BM3D denoiser was introduced in "Image denoising by sparse 3D transform-domain collaborative filtering", by
    Davob et al., IEEE Transactions on Image Processing (2007).


    """

    def __init__(self):
        super(BM3D, self).__init__()

    def forward(self, x, sigma):
        r"""
        Run the denoiser on image with noise level :math:`\sigma`.

        :param torch.Tensor x: noisy image
        :param float sigma: noise level (not used)
        """

        out = torch.zeros_like(x)

        for i in range(x.shape[0]):
            out[i, :, :, :] = array2tensor(
                bm3d.bm3d(tensor2array(x[i, :, :, :]), sigma)
            )
        return out


def tensor2array(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1)
