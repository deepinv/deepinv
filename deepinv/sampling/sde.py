import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable


class DiffusionSDE(nn.Module):
    def __init__(self, f: Callable, g: Callable, score: Callable, T: float = 1.0):
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

    def backward_sde(self, x: Tensor, t: Tensor) -> Tensor:
        pass
