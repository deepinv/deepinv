from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from deepinv.optim.data_fidelity import PoissonLikelihood
from deepinv.optim.utils import objective_function
from deepinv.utils.tensorlist import ones_like

from .optim_iterator import OptimIterator

if TYPE_CHECKING:
    from deepinv.optim import DataFidelity, Prior
    from deepinv.physics import Physics
    from deepinv.utils import TensorList


class OSEMIteration(OptimIterator):
    r"""
    Iterator for the Ordered-Subsets Expectation-Maximization (OSEM) algorithm.

    One iteration corresponds to a complete OSEM epoch, applying one
    multiplicative update for each measurement and physics subset.
    More details can be found in the documentation of the
    :class:`deepinv.optim.optimizers.OSEM` optimizer.
    """

    def __init__(self, eps: float = 1e-6, cost_fn=None, **kwargs):
        self.eps = eps
        super(OSEMIteration, self).__init__(cost_fn=None, **kwargs)
        self.cost_fn = cost_fn

    def forward(
        self,
        X: dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None],
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior | None,
        cur_params: dict,
        y: torch.Tensor | TensorList,
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
        :param torch.Tensor, deepinv.utils.TensorList y: Full input data or pre-split measurements.
        :param deepinv.physics.Physics physics: Full physics or pre-split stacked physics modeling the data-fidelity term.
        :param deepinv.utils.TensorList y_subsets: Measurement subsets.
        :param deepinv.physics.StackedLinearPhysics subset_physics: Physics operators corresponding to the measurement subsets.
        :return: Dictionary ``{"est": (x, None), "cost": F, "it": k + 1}`` containing the updated iterate and estimated cost.
        """
        x = X["est"][0]
        k = 0 if "it" not in X else X["it"]

        y_subsets = kwargs["y_subsets"]
        if isinstance(y, list) or isinstance(y_subsets, list):
            raise TypeError(
                "OSEMIteration requires pre-split measurements as a "
                "deepinv.utils.TensorList."
            )
        subset_physics = kwargs["subset_physics"]
        num_subsets = len(subset_physics)
        if num_subsets < 2:
            raise ValueError(
                "OSEM requires at least two subsets. " "Use deepinv.optim.MLEM instead."
            )
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

        # Since we support both pre-split and full physics / measurements,
        # the cost computation logic must branch to handle both cases.
        F = None
        if self.has_cost and cur_prior is not None:
            from deepinv.physics.forward import StackedLinearPhysics

            cost_fn = self.cost_fn or objective_function
            if isinstance(physics, StackedLinearPhysics):
                # For pre-split inputs, evaluate each subset objective and sum them.
                F = 0
                for cur_y, cur_physics in zip(y, physics, strict=True):
                    subset_data_fidelity = cur_data_fidelity
                    if (
                        isinstance(cur_data_fidelity, PoissonLikelihood)
                        and isinstance(cur_data_fidelity.bkg, torch.Tensor)
                        and cur_data_fidelity.bkg.numel() > 1
                        and hasattr(cur_physics, "background")
                    ):
                        subset_data_fidelity = PoissonLikelihood(
                            gain=cur_data_fidelity.gain,
                            bkg=cur_physics.background / cur_data_fidelity.gain,
                            denormalize=cur_data_fidelity.d.denormalize,
                        )
                    F = F + cost_fn(
                        x,
                        subset_data_fidelity,
                        cur_prior,
                        cur_params,
                        cur_y,
                        cur_physics,
                    )
            else:
                # Full inputs are evaluated directly in a single call.
                F = cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)

        return {"est": (x, None), "cost": F, "it": k + 1}
