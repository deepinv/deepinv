from __future__ import annotations

import torch
import torch.nn.functional as F

from .optim_iterator import OptimIterator
import deepinv.physics.functional as dF
from deepinv.optim.prior import ZeroPrior


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
        super().__init__(**kwargs)
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
            ``lambda_reg_x``, ``lambda_reg_k``, ``g_param`` and
            ``g_param_kernel``.
        :param torch.Tensor y: Blurry observation of shape ``(B, C, H, W)``.
        :param deepinv.physics.Physics physics: Blur physics updated in-place with the
            current kernel for the image update.
        """
        x_prev, k_prev = X["est"][:2]
        x = x_prev.clamp_min(self.eps)

        if y.dim() != 4 or x.dim() != 4:
            raise ValueError(
                "BlindRLIteration currently supports 2D images shaped (B, C, H, W)."
            )
        if y.shape != x.shape:
            raise ValueError(
                "BlindRLIteration requires circular deconvolution with y and x having the same shape."
                f"Got y={tuple(y.shape)} and x={tuple(x.shape)}."
            )

        k = k_prev.to(device=x.device, dtype=x.dtype)
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
        if self.normalize_kernel:
            k = F.normalize(
                k.clamp_min(0.0).flatten(1), p=1, dim=1, eps=self.eps
            ).view_as(k)
        else:
            k = k.clamp_min(0.0)

        hk, wk = k.shape[-2:]
        x_steps = int(cur_params.get("x_steps", 1))
        k_steps = int(cur_params.get("k_steps", 1))
        lambda_x = cur_params.get("lambda_reg_x", 0.0)
        lambda_k = cur_params.get("lambda_reg_k", 0.0)
        g_param = cur_params.get("g_param", None)
        k_g_param = cur_params.get("g_param_kernel", None)

        ones_y = torch.ones_like(y)

        # Kernel update: adjoint of A_x: h -> x * h with respect to h.
        # This is not physics.A_adjoint, which is the adjoint of A_h with
        # respect to the image x for a fixed kernel h.
        sensitivity_k = dF.conv2d_filter_adjoint(x, ones_y, (hk, wk)).mean(
            dim=1, keepdim=True
        )

        for _ in range(k_steps):
            y_hat = dF.conv2d(x, k, padding="circular")
            ratio = y / y_hat.clamp_min(self.eps)
            numerator_k = dF.conv2d_filter_adjoint(x, ratio, (hk, wk)).mean(
                dim=1, keepdim=True
            )
            denom_k = sensitivity_k + lambda_k * self.k_prior.grad(k, k_g_param)
            k = k * numerator_k / denom_k.clamp_min(self.eps)
            if self.normalize_kernel:
                k = F.normalize(
                    k.clamp_min(0.0).flatten(1), p=1, dim=1, eps=self.eps
                ).view_as(k)
            else:
                k = k.clamp_min(0.0)

        physics.update_parameters(filter=k)
        sensitivity_x = physics.A_adjoint(ones_y).clamp_min(self.eps)

        for _ in range(x_steps):
            y_hat = physics.A(x)
            numerator_x = physics.A_adjoint(y / y_hat.clamp_min(self.eps))
            prior_grad_x = cur_prior.grad(x, g_param)
            # Add a safeguard to avoid NaN values in the prior gradient,
            # which can occur often with non-smooth priors like TV or L1.
            prior_grad_x = torch.where(
                torch.isfinite(prior_grad_x),
                prior_grad_x,
                torch.zeros_like(prior_grad_x),
            )
            denom_x = sensitivity_x + lambda_x * prior_grad_x
            x = x * numerator_x / denom_x.clamp_min(self.eps)
            x = x.clamp_min(self.eps)

        k_it = 0 if "it" not in X else X["it"]
        cost = (
            self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.cost_fn is not None
            and self.has_cost
            and cur_data_fidelity is not None
            and cur_prior is not None
            else None
        )
        return {"est": (x, k), "cost": cost, "it": k_it + 1}
