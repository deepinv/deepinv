import torch
from torch import Tensor


class Welford:
    r"""
    Welford's algorithm :footcite:ps:`welford1962note`for calculating mean and variance.

    |sep|
    
    :References:
        .. footbibliography::
    """

    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = torch.zeros_like(x)

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S = self.S + (x - self.M) * (x - Mnext)
        self.M = Mnext

    def mean(self):
        return self.M

    def var(self):
        return self.S / (self.k - 1)


def refl_projbox(x, lower: Tensor, upper: Tensor) -> Tensor:
    x = torch.abs(x)
    return torch.clamp(x, min=lower, max=upper)


def projbox(x, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.clamp(x, min=lower, max=upper)
