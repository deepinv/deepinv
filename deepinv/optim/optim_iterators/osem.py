from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from deepinv.optim.data_fidelity import (
    PoissonLikelihood,
    StackedPhysicsDataFidelity,
)
from deepinv.optim.utils import objective_function
from deepinv.utils.tensorlist import ones_like

from .optim_iterator import OptimIterator

if TYPE_CHECKING:
    from deepinv.optim import DataFidelity, Prior
    from deepinv.physics import StackedLinearPhysics
    from deepinv.utils import TensorList


class OSEMIteration(OptimIterator):
    r"""
    Performs a single iteration of the OSEM algorithm :footcite:p:`hudsonAcceleratedImageReconstruction1994`,
    which is a classic baseline reconstruction method for inverse problems with Poisson noise statistics.
    Note that :class:`deepinv.optim.optim_iterators.MLEMIteration` is a special case with one subset only.
    More details on the algorithm can be found in the documentation of the
    :class:`deepinv.optim.optimizers.OSEM` optimizer.
    """

    def __init__(self, eps: float = 1e-6, cost_fn=None, **kwargs):
        super(OSEMIteration, self).__init__(cost_fn=None, **kwargs)
        self.eps = eps
        self._cost_fn = cost_fn or objective_function
        self.cost_fn = self._compute_cost

    def _compute_cost(
        self,
        x: torch.Tensor,
        data_fidelity: DataFidelity,
        prior: Prior,
        params: dict,
        y: TensorList,
        physics: StackedLinearPhysics,
    ) -> torch.Tensor:
        subset_data_fidelities = []
        for cur_physics in physics:
            subset_data_fidelity = data_fidelity
            if isinstance(data_fidelity, PoissonLikelihood) and hasattr(
                cur_physics, "background"
            ):
                subset_data_fidelity = PoissonLikelihood(
                    gain=data_fidelity.gain,
                    bkg=cur_physics.background / data_fidelity.gain,
                    denormalize=data_fidelity.d.denormalize,
                )
            subset_data_fidelities.append(subset_data_fidelity)
        return self._cost_fn(
            x,
            StackedPhysicsDataFidelity(subset_data_fidelities),
            prior,
            params,
            y,
            physics,
        )

    def forward(
        self,
        X: dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None],
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior | None,
        cur_params: dict,
        y: TensorList,
        physics: StackedLinearPhysics,
        *args,
        **kwargs,
    ) -> dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None]:
        r"""
        Perform one Ordered-Subsets Expectation-Maximization epoch.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param deepinv.utils.TensorList y: Measurement subsets.
        :param deepinv.physics.StackedLinearPhysics physics: Physics operators corresponding to the measurement subsets.
        :return: Dictionary ``{"est": (x, None), "cost": F, "it": k + 1}`` containing the updated iterate and estimated cost.
        """
        x = X["est"][0]
        k = 0 if "it" not in X else X["it"]

        num_subsets = len(physics)

        prior_scale = 1.0 / num_subsets
        for cur_y, cur_physics in zip(y, physics, strict=True):
            sensitivity = cur_physics.A_adjoint(ones_like(cur_y))
            # For deepinv.physics.PET, we need to add the background term
            if hasattr(cur_physics, "background"):
                proj = cur_physics.A(x, add_background=True)
            else:
                proj = cur_physics.A(x)

            numerator = x * cur_physics.A_adjoint(cur_y / proj.clamp(min=self.eps))
            denom = sensitivity

            # Scale the OSL prior so that one full epoch applies its total weight.
            if cur_prior is not None:
                prior_grad = cur_prior.grad(x, cur_params["g_param"])
                denom = denom + prior_scale * cur_params["lambda"] * prior_grad

            x = numerator / denom.clamp(min=self.eps)

        F = None
        if self.has_cost and cur_prior is not None:
            F = self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)

        return {"est": (x, None), "cost": F, "it": k + 1}
