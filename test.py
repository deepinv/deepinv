import torch
import deepinv as dinv

x = torch.randn(5, 3, 64, 64)
denoiser = dinv.models.WaveletDenoiser(level=3, wv="db8")
sigma = torch.ones(3, 3)
y = denoiser.prox_l1(x, ths=0.1)
