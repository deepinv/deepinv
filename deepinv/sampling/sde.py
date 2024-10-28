
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
        score: Callable = None,
        T: float = 1.0,
    ):
        super().__init__()
        self.f = f
        self.g = g
        self.score = score
        self.T = T

    def forward_sde(self, x: Tensor, num_steps: int = 100) -> Tensor:
        stepsize = self.T / num_steps
        for k in range(num_steps):
            t = k * stepsize
            dw = torch.randn_like(x)
            drift = self.f(x, t)
            diffusion = self.g(t)
            x = x + stepsize * drift + diffusion * np.sqrt(stepsize) * dw
        return x

    def backward_sde(
        self, x: Tensor, num_steps: int = 1000, alpha: float = 1.0
    ) -> Tensor:
        stepsize = self.T / num_steps
        t = 0
        for k in range(num_steps):
            rt = self.T - t * k
            g = self.g(rt)
            drift = self.f(x, rt) - (1 + alpha**2) * g**2 * self.score(x, rt)
            diffusion = alpha * g
            dw = torch.randn_like(x) 
            x = x + drift * stepsize + diffusion * np.sqrt(stepsize) * dw
        return x


if __name__ == "__main__":
    import deepinv as dinv

    from deepinv.utils.demo import load_url_image, get_image_url

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url, grayscale=False).to(device)

    score = dinv.models.WaveletDenoiser(wv="db8", level=4, device=device)
    OUSDE = DiffusionSDE(score=score, T=1.0)

    sample = OUSDE.forward_sde(x)

    # x = torch.randn((2, 1, 28, 28), device=device)
    # sample = OUSDE.backward_sde(x)
    dinv.utils.plot([x, sample])
