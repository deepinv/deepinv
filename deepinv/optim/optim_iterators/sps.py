from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .optim_iterator import OptimIterator

if TYPE_CHECKING:
    from deepinv.optim import Prior, StackedPhysicsDataFidelity
    from deepinv.physics import StackedLinearPhysics
    from deepinv.utils import TensorList


class SPSIteration(OptimIterator):
    r"""One SPS iteration, or one ordered-subsets transmission (OSTR) epoch.

    See :class:`deepinv.optim.SPS` for the objective and assumptions.
    """

    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def forward(
        self,
        X: dict,
        cur_data_fidelity: StackedPhysicsDataFidelity,
        cur_prior: Prior,
        cur_params: dict,
        y: TensorList,
        physics: StackedLinearPhysics,
        row_sums: list[torch.Tensor],
        **kwargs,
    ) -> dict:
        x = X["est"][0].clamp_min(0)
        for measurement, operator, row_sum, fidelity in zip(
            y, physics, row_sums, cur_data_fidelity.data_fidelity_list, strict=True
        ):
            mu = fidelity.mu
            line_integrals = mu * operator.A(x)
            predicted = torch.exp(-line_integrals)
            observed = torch.exp(-mu * measurement)

            # Optimal curvature for h(t) = exp(-t) + observed * t. Use its
            # Taylor expansion near zero to avoid cancellation and division by zero.
            t = line_integrals
            safe_t = t.clamp_min(1e-3)
            curvature = 2 * (-torch.expm1(-safe_t) - safe_t * torch.exp(-safe_t))
            curvature = curvature / safe_t.square()
            curvature = torch.where(t < 1e-3, 1 - 2 * t / 3 + t.square() / 4, curvature)

            numerator = operator.A_adjoint(predicted - observed)
            denominator = operator.A_adjoint(row_sum * curvature)
            min_curvature = max(self.eps, torch.finfo(denominator.dtype).tiny)
            x = (
                x
                + cur_params["stepsize"]
                * (numerator / denominator.clamp_min(min_curvature))
                / mu
            ).clamp_min(0)

        cost = (
            self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost and self.cost_fn is not None
            else None
        )
        return {"est": (x, None), "cost": cost, "it": X.get("it", 0) + 1}
