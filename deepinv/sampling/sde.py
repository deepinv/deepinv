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
        stepsize = 1.0 / num_steps
        for t in range(num_steps):
            dw      = torch.randn_like(x_new)
            f_dt    = self.f(x_new, t)
            g_dw    = self.g(t) * dw
            x_new   = x_new + stepsize*(f_dt + g_dw)
        return x_new

    def prior_sampling(self, batch_size: int = 1) -> Tensor:
        # return torch.randn((batch_size, ))
        pass

    def backward_sde(
        self, x: Tensor, num_steps: int = 1000, alpha: float = 1.0
    ) -> Tensor:
        dt = self.T / num_steps
        t = 0
        for i in range(num_steps):
            rt = self.T - t
            g = self.g(x, rt)
            drift = self.f(x, rt) - (1 + alpha**2) * g**2 * self.score(x, rt)
            diffusion = alpha * g
