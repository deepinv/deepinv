
import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable
import numpy as np


class DiffusionSDE(nn.Module):
    def __init__(
        self,
        f: Callable = lambda x, t: -x,
        g: Callable = lambda t: math.sqrt(2.0),
        prior: Callable = None,
        T: float = 1.0,
    ):
        super().__init__()
        self.T = T
        self.drift_forw = lambda x,t : f(x, t)
        self.diff_forw = lambda t : g(t)
        self.drift_back = lambda x,t,alpha : - f(x, T-t) - (1 + alpha**2) * prior.grad(x, T-t) * g(T-t)**2
        self.diff_back = lambda t, alpha : alpha * g(T-t)

    def forward_sde(self, x: Tensor, num_steps: int = 100) -> Tensor:
        stepsize = self.T / num_steps
        for k in range(num_steps):
            t = k * stepsize
            x += stepsize * self.drift_forw(x,t) + self.diff_forw(t) * np.sqrt(stepsize) * torch.randn_like(x)
        return x

    def backward_sde(
        self, x: Tensor, num_steps: int = 100, alpha: float = 1.0
    ) -> Tensor:
        stepsize = self.T / num_steps
        for k in range(num_steps):
            t = k * stepsize
            x += stepsize * self.drift_back(x,t,alpha) + self.diff_back(t,alpha) * np.sqrt(stepsize) * torch.randn_like(x)
        return x


if __name__ == "__main__":
    import deepinv as dinv
    from deepinv.utils.demo import load_url_image, get_image_url

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url, grayscale=False).to(device)

    denoiser = dinv.models.WaveletDenoiser(wv="db8", level=4, device=device)
    prior = dinv.optim.prior.ScorePrior(denoiser = denoiser)
    
    OUSDE = DiffusionSDE(prior=prior, T=1.0)
    sample_noise = OUSDE.forward_sde(x)
    sample = OUSDE.backward_sde(torch.randn_like(x))
    dinv.utils.plot([x, sample_noise, sample])
