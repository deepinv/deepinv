
import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable


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

    def forward_sde(self, x: Tensor, num_steps: int) -> Tensor:
        x_new = x
        stepsize = self.T / num_steps
        for k in range(num_steps):
            t = k * stepsize
            dw = torch.randn_like(x_new)
            f_dt = self.f(x_new, t)
            g_dw = self.g(t) * dw
            x_new = x_new + stepsize * (f_dt + g_dw)
        return x_new

    def backward_sde(
        self, x: Tensor, num_steps: int = 1000, alpha: float = 1.0
    ) -> Tensor:
        dt = self.T / num_steps
        t = 0
        for k in range(num_steps):
            rt = self.T - t * k
            g = self.g(rt)
            drift = self.f(x, rt) - (1 + alpha**2) * g**2 * self.score(x, rt)
            diffusion = alpha * g
            dw = torch.randn_like(x) * dt
            x = x + drift * dt + diffusion * dw
        return x


if __name__ == "__main__":
    import deepinv as dinv

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    score = dinv.models.WaveletDenoiser(wv="db8", level=4, device=device)
    OUSDE = DiffusionSDE(score=score, T=1.0)

    x = torch.randn((2, 1, 28, 28), device=device)
    sample = OUSDE.backward_sde(x)

    dinv.utils.plot([x, sample])
