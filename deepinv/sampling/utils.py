import torch
from torch import Tensor


class Welford:
    r"""
    Welford's algorithm for calculating mean and variance

    https://doi.org/10.2307/1266577
    """

    def __init__(self, x: Tensor):
        self.k = 1
        self.M = x.clone()
        self.S = torch.zeros_like(x)

    def update(self, x: Tensor):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S = self.S + (x - self.M) * (x - Mnext)
        self.M = Mnext

    def mean(self) -> Tensor:
        return self.M

    def var(self) -> Tensor:
        if self.k > 1:
            return self.S / (self.k - 1)
        else:
            return self.S


def refl_projbox(x, lower: Tensor, upper: Tensor) -> Tensor:
    x = torch.abs(x)
    return torch.clamp(x, min=lower, max=upper)


def projbox(x, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.clamp(x, min=lower, max=upper)
