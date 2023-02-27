import math
import numpy as np
import torch
import torch.nn as nn

from .denoiser import register

from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)

@register('waveletprior')
class WaveletPrior(nn.Module):
    def __init__(self, level=3, wv='db8'):
        super().__init__()
        self.level = level
        self.dwt = DWTForward(J=self.level, wave=wv)
        self.iwt = DWTInverse(wave=wv)

    def prox_l1(self, x, ths=0.1):
        return torch.maximum(torch.Tensor([0]).type(x.dtype), x - ths) + torch.minimum(torch.Tensor([0]).type(x.dtype),
                                                                                       x + ths)

    def forward(self, x, ths=0.):
        coeffs = self.dwt(x)
        for l in range(self.level):
            coeffs[1][l] = self.prox_l1(coeffs[1][l], ths)
        y = self.iwt(coeffs)
        return y


@register('waveletdictprior')
class WaveletDict(nn.Module):
    def __init__(self, level=3, list_wv=['db8', 'db4'], max_iter=10):
        super().__init__()
        self.level = level
        self.list_prox = nn.ModuleList([WaveletPrior(level=level, wv=wv) for wv in list_wv])
        self.max_iter = max_iter

    def forward(self, y, ths=0.):
        z_p = y.repeat(len(self.list_prox), 1, 1, 1, 1)
        p_p = torch.zeros_like(z_p)
        x = p_p.clone()
        for it in range(self.max_iter):
            x_prev = x.clone()
            for p in range(len(self.list_prox)):
                p_p[p, ...] = self.list_prox[p](z_p[p, ...], ths)
            x = p_p.mean(axis=0)
            for p in range(len(self.list_prox)):
                z_p[p, ...] = x + z_p[p, ...] - p_p[p, ...]
            rel_crit = torch.linalg.norm((x - x_prev).flatten()) / torch.linalg.norm(x.flatten() + 1e-6)
            if rel_crit<1e-3:
                break
        return x
