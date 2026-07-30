from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from deepinv.optim.data_fidelity import StackedPhysicsDataFidelity
from deepinv.optim.utils import objective_function
from deepinv.utils.tensorlist import ones_like

from .optim_iterator import OptimIterator

if TYPE_CHECKING:
    from deepinv.optim import DataFidelity, Prior
    from deepinv.physics import Physics
    from deepinv.utils import TensorList


def _osem_objective_function(
    x: torch.Tensor,
    data_fidelity: DataFidelity,
    prior: Prior | None,
    cur_params: dict,
    y: torch.Tensor | TensorList | list[torch.Tensor],
    physics: Physics,
) -> torch.Tensor:
    """Evaluate the OSEM objective for full or pre-split inputs."""
    from deepinv.physics.forward import StackedLinearPhysics

    if not isinstance(physics, StackedLinearPhysics):
        return objective_function(x, data_fidelity, prior, cur_params, y, physics)

    if isinstance(data_fidelity, StackedPhysicsDataFidelity):
        data_term = data_fidelity(x, y, physics)
    else:
        data_term = sum(
            data_fidelity(x, cur_y, cur_physics)
            for cur_y, cur_physics in zip(y, physics, strict=True)
        )

    if prior is not None and prior.explicit_prior:
        return data_term + cur_params["lambda"] * prior(x, cur_params["g_param"])
    return data_term


class OSEMIteration(OptimIterator):
    r"""
    Iterator for the Ordered-Subsets Expectation-Maximization (OSEM) algorithm.

    One iteration corresponds to a complete OSEM epoch, applying one
    multiplicative update for each measurement and physics subset.
    More details can be found in the documentation of the
    :class:`deepinv.optim.optimizers.OSEM` optimizer.
    """

    def __init__(self, eps: float = 1e-15, cost_fn=None, **kwargs):
        self.eps = eps
        super(OSEMIteration, self).__init__(
            cost_fn=_osem_objective_function if cost_fn is None else cost_fn,
            **kwargs,
        )

    def forward(
        self,
        X: dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None],
        cur_data_fidelity: DataFidelity | None,
        cur_prior: Prior | None,
        cur_params: dict,
        y: torch.Tensor | TensorList | list[torch.Tensor],
        physics: Physics,
        *args,
        **kwargs,
    ) -> dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None]:
        r"""
        Perform one Ordered-Subsets Expectation-Maximization epoch.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor, deepinv.utils.TensorList, list[torch.Tensor] y: Full input data or pre-split measurements.
        :param deepinv.physics.Physics physics: Full physics or pre-split stacked physics modeling the data-fidelity term.
        :param deepinv.utils.TensorList y_subsets: Measurement subsets.
        :param deepinv.physics.StackedLinearPhysics subset_physics: Physics operators corresponding to the measurement subsets.
        :return: Dictionary ``{"est": (x, None), "cost": F, "it": k + 1}`` containing the updated iterate and estimated cost.
        """
        x = X["est"][0]
        k = 0 if "it" not in X else X["it"]

        y_subsets = kwargs["y_subsets"]
        subset_physics = kwargs["subset_physics"]
        num_subsets = len(subset_physics)
        if num_subsets < 1:
            raise ValueError("OSEM requires at least one subset.")
        if len(y_subsets) != num_subsets:
            raise ValueError(
                "The number of measurement subsets and physics subsets must match."
            )

        prior_scale = 1.0 / num_subsets
        for cur_y, cur_physics in zip(y_subsets, subset_physics, strict=True):
            sensitivity = cur_physics.A_adjoint(ones_like(cur_y))
            # For deepinv.physics.PET, we need to add the background term
            if hasattr(cur_physics, "background"):
                proj = cur_physics.A(x, add_background=True)
            # Other deepinv.physics.Physics do not have a background term
            else:
                proj = cur_physics.A(x)

            numerator = x * cur_physics.A_adjoint(cur_y / proj.clamp(min=self.eps))
            denom = sensitivity

            # Scale the OSL prior so that one full epoch applies its total weight.
            if cur_prior is not None:
                prior_grad = cur_prior.grad(x, cur_params["g_param"])
                denom = denom + prior_scale * cur_params["lambda"] * prior_grad

            x = numerator / denom.clamp(min=self.eps)

        F = (
            self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.cost_fn is not None
            and self.has_cost
            and cur_data_fidelity is not None
            and cur_prior is not None
            else None
        )
        return {"est": (x, None), "cost": F, "it": k + 1}
