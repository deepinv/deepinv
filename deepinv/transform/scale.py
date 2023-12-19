import torch
import torch.nn.functional as F
from torch.nn import Module


def sample_from(values, shape=(1,), dtype=torch.float32, device="cpu"):
    """Sample a random tensor from a list of values"""
    values = torch.tensor(values, device=device, dtype=dtype)
    N = torch.tensor(len(values), device=device, dtype=dtype)
    indices = torch.floor(N * torch.rand(shape, device=device, dtype=dtype)).to(
        torch.int
    )
    return values[indices]


class Scale(Module):
    """
    2D Scaling
    See https://arxiv.org/abs/2312.11232
    Code from https://github.com/jscanvic/Scale-Equivariant-Imaging

    :param list factors: list of scale factors (default: [.75, .5])
    :param str padding_mode: padding mode for grid sampling
    :param str mode: interpolation mode for grid sampling
    """

    def __init__(self, factors=None, padding_mode="reflection", mode="bicubic"):
        super().__init__()

        self.factors = factors or [0.75, 0.5]
        self.padding_mode = padding_mode
        self.mode = mode

    def forward(self, x):
        b, _, h, w = x.shape

        # Sample a random scale factor for each batch element
        factor = sample_from(self.factors, shape=(b,), device=x.device)
        factor = factor.view(b, 1, 1, 1).repeat(1, 1, 1, 2)

        # Sample a random transformation center for each batch element
        # with coordinates in [-1, 1]
        center = torch.rand((b, 2), dtype=x.dtype, device=x.device)
        center = center.view(b, 1, 1, 2)
        center = 2 * center - 1

        # Compute the sampling grid for the scale transformation
        u = torch.arange(w, dtype=x.dtype, device=x.device)
        v = torch.arange(h, dtype=x.dtype, device=x.device)
        u = 2 / w * u - 1
        v = 2 / h * v - 1
        U, V = torch.meshgrid(u, v)
        grid = torch.stack([V, U], dim=-1)
        grid = grid.view(1, h, w, 2).repeat(b, 1, 1, 1)
        grid = 1 / factor * (grid - center) + center

        return F.grid_sample(
            x, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=True
        )
