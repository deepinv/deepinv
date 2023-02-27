import warnings
import bm3d

import torch
import torch.nn as nn
from .denoiser import register


@register('bm3d')
class BM3D(nn.Module):
    '''
    BM3D denoiser
    '''

    def __init__(self):
        super(BM3D, self).__init__()

    def forward(self, x, sigma):
        return torch.cat([array2tensor(bm3d.bm3d(tensor2array(xi), sigma)) for xi in x])


def tensor2array(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1)
