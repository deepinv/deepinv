from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from .optim_iterator import OptimIterator
import deepinv.physics.functional as dF
from deepinv.optim.prior import ZeroPrior

if TYPE_CHECKING:
    from deepinv.optim import DataFidelity, Prior
    from deepinv.physics import Physics


class BlindRLIteration(OptimIterator):
    r"""
    Iterator for Blind Richardson-Lucy deconvolution.

    This iterator performs one step to estimate the next kernel, and one step to
    estimate the next image.

    The current iterate is stored as ``X["est"] = (x, k)``. The kernel update
    assumes 2D circular convolution and a spatially invariant kernel shared by
    all image channels.

    :param deepinv.optim.Prior, None k_prior: optional kernel prior. Default: ``None``.
    :param bool normalize_kernel: whether to normalize the kernel to unit sum. Default: ``True``.
    :param bool use_fft: whether to use the FFT implementations for convolutions. Default: ``False``.
    :param float eps: numerical stability constant used for divisions. Default: ``1e-8``.
    """

    def __init__(
        self,
        k_prior: Prior = None,
        normalize_kernel: bool = True,
        use_fft: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k_prior = ZeroPrior() if k_prior is None else k_prior
        self.normalize_kernel = normalize_kernel
        self.use_fft = use_fft
        self.eps = eps

    def forward(
        self,
        X: dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor],
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        cur_params: dict,
        y: torch.Tensor,
        physics: Physics,
        *args,
        **kwargs,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        r"""
        Single Blind Richardson-Lucy iteration.

        :param dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor] X: Current
            iterate with ``X["est"] = (x, k)``.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Data fidelity term.
        :param deepinv.optim.Prior cur_prior: Image prior.
        :param dict cur_params: Parameters containing ``x_steps``, ``k_steps``,
            ``lambda_reg_x``, ``lambda_reg_k``, ``g_param`` and
            ``g_param_kernel``.
        :param torch.Tensor y: Blurry observation of shape ``(B, C, H, W)``.
        :param deepinv.physics.Physics physics: Blur physics updated in-place with the
            current kernel for the image update.
        :return: Dictionary ``{"est": (x, k), "cost": F, "it": it}`` containing
            the updated image, kernel, cost, and iteration number.
        """

        x_prev, k_prev = X["est"][:2]
        x = x_prev.clamp_min(self.eps)
        k = k_prev

        if self.normalize_kernel:
            k = F.normalize(k.flatten(1), p=1, dim=1, eps=self.eps).view_as(k)

        hk, wk = k.shape[-2:]
        x_steps = cur_params.get("x_steps", 1)
        k_steps = cur_params.get("k_steps", 1)
        lambda_x = cur_params.get("lambda_reg_x", 0.0)
        lambda_k = cur_params.get("lambda_reg_k", 0.0)
        g_param = cur_params.get("g_param", None)
        k_g_param = cur_params.get("g_param_kernel", None)

        ones_y = torch.ones_like(y)

        # Kernel update
        filter_adjoint = (
            dF.conv_filter_transpose2d
            if self.use_fft
            else dF.conv_filter_transpose2d_fft
        )
        sensitivity_k = filter_adjoint(x, ones_y, (hk, wk), padding="circular").sum(
            dim=1, keepdim=True
        )

        for _ in range(k_steps):
            y_hat = dF.conv2d(x, k, padding="circular")
            ratio = y / y_hat.clamp_min(self.eps)
            numerator_k = filter_adjoint(x, ratio, (hk, wk), padding="circular").sum(
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

        # Image update
        for _ in range(x_steps):
            y_hat = physics.A(x)
            numerator_x = physics.A_adjoint(y / y_hat.clamp_min(self.eps))
            denom_x = sensitivity_x + lambda_x * cur_prior.grad(x, g_param)
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
