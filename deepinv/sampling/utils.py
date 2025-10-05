import torch
from torch import Tensor


class Welford:
    r"""
    Welford's algorithm :footcite:t:`welford1962note`for calculating mean and variance.


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


def trapz_torch(
    func: Callable[[Tensor], Tensor], a: Tensor, b: Tensor, n_steps: int = 1000
) -> Tensor:
    """
    Differentiable trapezoidal integration of func(x) between a and b along the last dimension.

    :param func: Callable that takes a Tensor of shape (..., N) and returns Tensor of same shape.
    :param torch.Tensor a: Lower integration bound. Can be scalar or batched (...,).
    :param torch.Tensor b: Upper integration bound. Must be broadcastable with a.
    :param int n_steps: Number of discretization steps.
    :return torch.Tensor: Integral of func from a to b. Shape (...), integration along the last dimension.
    """
    # Create uniform grid between a and b
    shape = a.shape
    t = torch.linspace(0, 1, n_steps, device=b.device, dtype=b.dtype)
    # (..., 1) + (..., 1) * (1D linspace) => (..., N)
    x = a.unsqueeze(-1) + (b - a).unsqueeze(-1) * t

    # Evaluate function
    y = func(x)

    # Trapezoidal integration along the last dimension
    dx = x[..., 1:] - x[..., :-1]  # (..., N-1)
    area = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    return area.sum(dim=-1)
