import torch

class Welford:
    """
    Welford's online algorithm for calculating mean and variance.

    https://doi.org/10.2307/1266577
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
        return self.S / (self.k - 1) if self.k > 1 else self.S


