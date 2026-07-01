from __future__ import annotations

import torch
import torch.nn.functional as F

from .optim_iterator import OptimIterator
from deepinv.optim.prior import ZeroPrior


def _normalize_kernel(k: torch.Tensor, eps: float) -> torch.Tensor:
    """Project nonnegative kernels to unit sum per batch."""
    k = k.clamp_min(0.0)
    return k / k.sum(dim=tuple(range(1, k.dim())), keepdim=True).clamp_min(eps)


def _prepare_kernel(
    k: torch.Tensor,
    x: torch.Tensor,
    normalize: bool,
    eps: float,
) -> torch.Tensor:
    if k.dim() != 4:
        raise ValueError(
            "The blur kernel must be a 4D tensor shaped (B or 1, C or 1, H, W)."
        )

    if k.shape[0] == 1 and x.shape[0] > 1:
        k = k.expand(x.shape[0], -1, -1, -1).contiguous()
    elif k.shape[0] != x.shape[0]:
        raise ValueError(
            "The kernel batch size must be 1 or match x. "
            f"Got {k.shape[0]} and {x.shape[0]}."
        )

    if k.shape[1] == x.shape[1]:
        k = k.mean(dim=1, keepdim=True)
    elif k.shape[1] != 1:
        raise ValueError(
            "The kernel channel size must be 1 or match x. "
            f"Got {k.shape[1]} and {x.shape[1]}."
        )

    k = k.to(device=x.device, dtype=x.dtype)
    return _normalize_kernel(k, eps) if normalize else k.clamp_min(0.0)


def _expand_kernel_channels(k: torch.Tensor, channels: int) -> torch.Tensor:
    if k.shape[1] == 1 and channels > 1:
        return k.expand(-1, channels, -1, -1)
    return k


def _circular_patches(x: torch.Tensor, kernel_size: tuple[int, int]) -> torch.Tensor:
    h, w = kernel_size
    ph, pw = h // 2, w // 2
    ih, iw = (h - 1) % 2, (w - 1) % 2
    x_pad = F.pad(x, (pw - iw, pw, ph - ih, ph), mode="circular")
    patches = F.unfold(x_pad, kernel_size=kernel_size)
    return patches.view(x.shape[0], x.shape[1], h * w, x.shape[-2] * x.shape[-1])


def _kernel_forward_from_patches(
    patches: torch.Tensor,
    k: torch.Tensor,
    out_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    b, c, h, w = out_shape
    hk, wk = k.shape[-2:]
    k_flip = _expand_kernel_channels(k.flip(-2, -1), c)
    return (patches * k_flip.reshape(b, c, hk * wk, 1)).sum(dim=2).view(b, c, h, w)


def _kernel_adjoint_from_patches(
    patches: torch.Tensor,
    y: torch.Tensor,
    kernel_size: tuple[int, int],
) -> torch.Tensor:
    hk, wk = kernel_size
    y_flat = y.reshape(y.shape[0], y.shape[1], 1, -1)
    return (
        (patches * y_flat).sum(dim=-1).view(y.shape[0], y.shape[1], hk, wk).flip(-2, -1)
    )


def _prior_grad(prior, x: torch.Tensor, g_param, eps: float) -> torch.Tensor:
    if hasattr(prior, "nabla") and hasattr(prior, "nabla_adjoint"):
        grad_x = prior.nabla(x)
        return prior.nabla_adjoint(grad_x / torch.abs(grad_x).clamp_min(eps))
    return prior.grad(x, g_param)


class BlindRLIteration(OptimIterator):
    r"""
    Iterator for Blind Richardson-Lucy deconvolution.

    This iterator alternates multiplicative MLEM updates for a nonnegative blur kernel
    and a nonnegative image under the Poisson model

    .. math::
        y \sim \operatorname{Poisson}(k * x).

    The current iterate is stored as ``X["est"] = (x, k)``. The kernel update
    assumes 2D circular convolution and a spatially invariant kernel shared by
    all image channels.
    """

    def __init__(
        self,
        k_prior=None,
        normalize_kernel: bool = True,
        eps: float = 1e-15,
        **kwargs,
    ):
        super().__init__(has_cost=False, cost_fn=None, **kwargs)
        self.k_prior = ZeroPrior() if k_prior is None else k_prior
        self.normalize_kernel = normalize_kernel
        self.eps = eps

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        Single Blind Richardson-Lucy iteration.

        :param dict X: Current iterate with ``X["est"] = (x, k)``.
        :param deepinv.optim.Prior cur_prior: Image prior.
        :param dict cur_params: Parameters containing ``x_steps``, ``k_steps``,
            ``lambda``, ``lambda_kernel``, ``g_param`` and ``g_param_kernel``.
        :param torch.Tensor y: Blurry observation of shape ``(B, C, H, W)``.
        :param deepinv.physics.Physics physics: Blur physics updated in-place with the
            current kernel for the image update.
        """
        x_prev, k_prev = X["est"][:2]
        x = x_prev.clamp_min(self.eps)
        k = _prepare_kernel(k_prev, x, self.normalize_kernel, self.eps)

        if y.dim() != 4 or x.dim() != 4:
            raise ValueError(
                "BlindRLIteration currently supports 2D images shaped (B, C, H, W)."
            )
        if y.shape != x.shape:
            raise ValueError(
                "BlindRLIteration requires circular deconvolution with y and x having "
                f"the same shape. Got y={tuple(y.shape)} and x={tuple(x.shape)}."
            )

        b, c, h, w = y.shape
        hk, wk = k.shape[-2:]
        x_steps = int(cur_params.get("x_steps", 1))
        k_steps = int(cur_params.get("k_steps", 1))
        lambda_x = cur_params.get("lambda", 0.0)
        lambda_k = cur_params.get("lambda_kernel", 0.0)
        g_param = cur_params.get("g_param", None)
        k_g_param = cur_params.get("g_param_kernel", None)

        ones_y = torch.ones_like(y)

        x_patches = _circular_patches(x, (hk, wk))
        sensitivity_k = _kernel_adjoint_from_patches(
            x_patches, ones_y, (hk, wk)
        ).mean(dim=1, keepdim=True)

        for _ in range(k_steps):
            y_hat = _kernel_forward_from_patches(x_patches, k, (b, c, h, w))
            ratio = y / y_hat.clamp_min(self.eps)
            numerator_k = _kernel_adjoint_from_patches(
                x_patches, ratio, (hk, wk)
            ).mean(dim=1, keepdim=True)
            denom_k = sensitivity_k + lambda_k * self.k_prior.grad(k, k_g_param)
            k = k * numerator_k / denom_k.clamp_min(self.eps)
            k = (
                _normalize_kernel(k, self.eps)
                if self.normalize_kernel
                else k.clamp_min(0.0)
            )

        physics.update_parameters(filter=_expand_kernel_channels(k, c))
        sensitivity_x = physics.A_adjoint(ones_y).clamp_min(self.eps)

        for _ in range(x_steps):
            y_hat = physics.A(x)
            numerator_x = physics.A_adjoint(y / y_hat.clamp_min(self.eps))
            denom_x = sensitivity_x + lambda_x * _prior_grad(
                cur_prior, x, g_param, self.eps
            )
            x = x * numerator_x / denom_x.clamp_min(self.eps)
            x = x.clamp_min(self.eps)

        k_it = 0 if "it" not in X else X["it"]
        return {"est": (x, k), "cost": None, "it": k_it + 1}
