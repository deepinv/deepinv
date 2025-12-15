from __future__ import annotations
import torch
from torch import Tensor
from .base import Denoiser


class BilateralFilter(Denoiser):
    r"""
    Bilateral filter.

    The bilateral filter as it was introduced in :footcite:t:`tomasi1998bilateral`.
    where each output pixel is a normalized weighted average of neighboring pixels in a spatial window.
    The weights factor into a spatial kernel and a range kernel:

        \hat{x}_i = (1 / W_i) * \sum_{j \in Ω(i)} k_d(i - j) k_r(x_i - x_j) x_j

    with Gaussian spatial kernel k_d and Gaussian range kernel k_r.
    The spatial standard deviation $\sigma_d$ controls how fast the kernel
    decays with distance, while the range standard deviation $\sigma_r$
    controls how strongly intensity differences are penalized.

    |sep|

    >>> import deepinv as dinv
    >>> x = dinv.utils.load_example("butterfly.png")
    >>> physics = dinv.physics.GaussianNoise()
    >>> y = physics(x)
    >>> model = dinv.models.BilateralFilter()
    >>> x_hat = model(y,sigma_d=1.1,sigma_r=0.3,window_size=9)
    >>> dinv.metric.PSNR()(x_hat, x) > 26.13
    tensor([True])
    """

    def forward(
        self,
        x: Tensor,
        sigma_d: float | Tensor = 1,
        sigma_r: float | Tensor = 1,
        window_size: int | Tensor = 5,
        **kwargs,
    ) -> Tensor:
        r"""
        Apply a bilateral filter to the input x.

        :param torch.Tensor x : Input tensor of shape (B, C, H, W) or (C, H, W).
        :param float sigma_d : Spatial standard deviation (spatial domain).
            Larger values yield larger effective receptive fields.
        :param float sigma_r : Range standard deviation (intensity domain).
            Larger values yield stronger smoothing across edges.
        :param int window_size :
            Size of the (square) spatial window. Must be an odd integer.

        :return: Filtered image.
        """

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")

        # Accept (C, H, W) by adding a batch dimension
        added_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            added_batch = True
        if x.dim() != 4:
            raise ValueError("Input must have shape (B, C, H, W) or (C, H, W).")

        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # sigma_d: scalar or shape (B,)
        if isinstance(sigma_d, torch.Tensor):
            if sigma_d.numel() == 1:
                sigma_d = sigma_d.to(device=device, dtype=dtype).view(1, 1, 1, 1, 1, 1)
            else:
                if sigma_d.numel() != B:
                    raise ValueError(
                        f"sigma_d tensor must have length B={B}, got {sigma_d.numel()}"
                    )
                sigma_d = sigma_d.to(device=device, dtype=dtype).view(B, 1, 1, 1, 1, 1)
        else:
            sigma_d = torch.tensor([float(sigma_d)], device=device, dtype=dtype).view(
                1, 1, 1, 1, 1, 1
            )

        # sigma_r: scalar or shape (B,)
        if isinstance(sigma_r, torch.Tensor):
            if sigma_r.numel() == 1:
                sigma_r = sigma_r.to(device=device, dtype=dtype).view(1, 1, 1, 1, 1, 1)
            else:
                if sigma_r.numel() != B:
                    raise ValueError(
                        f"sigma_r tensor must have length B={B}, got {sigma_r.numel()}"
                    )
                sigma_r = sigma_r.to(device=device, dtype=dtype).view(B, 1, 1, 1, 1, 1)
        else:
            sigma_r = torch.tensor([float(sigma_r)], device=device, dtype=dtype).view(
                1, 1, 1, 1, 1, 1
            )

        # Precompute spatial kernel (Gaussian) over the window
        half = window_size // 2
        coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        spatial_sq = xx**2 + yy**2  # shape (K, K)
        spatial_sq = spatial_sq.view(1, 1, 1, 1, window_size, window_size)

        spatial_kernel = torch.exp(-0.5 * spatial_sq / (sigma_d**2))

        # Extract sliding local patches of shape (B, C, H, W, K, K)
        patches = torch.nn.functional.unfold(
            x, kernel_size=window_size, padding=half
        )  # (B, C*K*K, H*W)
        patches = patches.view(B, C, window_size, window_size, H, W)
        patches = patches.permute(0, 1, 4, 5, 2, 3)  # (B, C, H, W, K, K)

        center = x.view(B, C, H, W, 1, 1)

        # Range kernel

        diff = patches - center  # (B, C, H, W, K, K)
        range_sq = diff.pow(2)
        range_kernel = torch.exp(-0.5 * range_sq / (sigma_r**2))

        weights = spatial_kernel * range_kernel  # (B, C, H, W, K, K)

        # Normalize weights over the window
        weights_sum = weights.sum(dim=(-1, -2), keepdim=True)  # (B, C, H, W, 1, 1)
        weights_sum = torch.clamp(weights_sum, min=1e-12)
        weights_norm = weights / weights_sum

        # Weighted average
        out = (weights_norm * patches).sum(dim=(-1, -2))  # (B, C, H, W)

        if added_batch:
            out = out.squeeze(0)

        return out
