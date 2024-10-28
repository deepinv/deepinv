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

    def forward_sde(self, x: Tensor, t: Tensor) -> Tensor:
        pass

    def backward_sde(self, x: Tensor, t: Tensor) -> Tensor:
        pass
