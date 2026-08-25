from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .optim_iterator import OptimIterator

if TYPE_CHECKING:
    from deepinv.optim import DataFidelity, Prior
    from deepinv.physics import StackedLinearPhysics
    from deepinv.utils import TensorList


class BSREMIteration(OptimIterator):
    r"""Perform one epoch of the BSREM algorithm.

    See :class:`deepinv.optim.BSREM` for the update equations and references.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        sensitivity_threshold: float = 1e-2,
        cost_fn=None,
        **kwargs,
    ):
        super().__init__(cost_fn=cost_fn, **kwargs)
        self.eps = eps
        self.sensitivity_threshold = sensitivity_threshold

    def forward(
        self,
        X: dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None],
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        cur_params: dict,
        y: TensorList,
        physics: StackedLinearPhysics,
        sensitivities: list[torch.Tensor],
        *args,
        **kwargs,
    ) -> dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None]:
        r"""Perform one Block Sequential Regularized EM epoch."""
        x = X["est"][0]
        k = 0 if "it" not in X else X["it"]
        num_subsets = len(physics)
        average_sensitivity = sum(sensitivities) / num_subsets
        preconditioner_denominator = average_sensitivity.clamp(min=self.eps)
        sensitivity_support = average_sensitivity > (
            self.sensitivity_threshold * average_sensitivity.amax()
        )

        for cur_y, cur_physics, cur_sensitivity in zip(
            y, physics, sensitivities, strict=True
        ):
            if hasattr(cur_physics, "background"):
                projection = cur_physics.A(x, add_background=True)
            else:
                projection = cur_physics.A(x)

            data_gradient = cur_sensitivity - cur_physics.A_adjoint(
                cur_y / projection.clamp(min=self.eps)
            )
            prior_gradient = (
                cur_params["lambda"]
                * cur_prior.grad(x, cur_params["g_param"])
                / num_subsets
            )
            preconditioner = torch.where(
                sensitivity_support,
                x / preconditioner_denominator,
                torch.zeros_like(x),
            )
            candidate = (
                x
                - cur_params["stepsize"]
                * preconditioner
                * (data_gradient + prior_gradient)
            ).clamp(min=self.eps)
            x = torch.where(
                sensitivity_support,
                candidate,
                torch.full_like(candidate, self.eps),
            )

        F = (
            self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"est": (x, None), "cost": F, "it": k + 1}
