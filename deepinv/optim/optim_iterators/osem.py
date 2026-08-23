from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .optim_iterator import OptimIterator

if TYPE_CHECKING:
    from deepinv.optim import DataFidelity, Prior
    from deepinv.physics import StackedLinearPhysics
    from deepinv.utils import TensorList


class OSEMIteration(OptimIterator):
    r"""
    Performs a single iteration of the OSEM algorithm, which is a classic baseline reconstruction method for inverse problems with Poisson noise statistics.
    Note that :class:`deepinv.optim.optim_iterators.MLEMIteration` is a special case with one subset only.
    More details on the algorithm can be found in the documentation of the
    :class:`deepinv.optim.optimizers.OSEM` optimizer.
    """

    def __init__(self, eps: float = 1e-6, cost_fn=None, **kwargs):
        super(OSEMIteration, self).__init__(cost_fn=cost_fn, **kwargs)
        self.eps = eps

    def forward(
        self,
        X: dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None],
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior | None,
        cur_params: dict,
        y: TensorList,
        physics: StackedLinearPhysics,
        sensitivities: list[torch.Tensor],
        *args,
        **kwargs,
    ) -> dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None]:
        r"""
        Perform one Ordered-Subsets Expectation-Maximization step.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param deepinv.utils.TensorList y: Measurement subsets.
        :param deepinv.physics.StackedLinearPhysics physics: Physics operators corresponding to the measurement subsets.
        :param list[torch.Tensor] sensitivities: Precomputed sensitivity maps :math:`A_l^T \mathbf{1}` for each subset.
        :return: Dictionary ``{"est": (x, None), "cost": F, "it": k + 1}`` containing the updated iterate and estimated cost.
        """
        x = X["est"][0]
        k = 0 if "it" not in X else X["it"]

        num_subsets = len(physics)

        prior_scale = 1.0 / num_subsets
        for cur_y, cur_physics, cur_sensitivity in zip(
            y, physics, sensitivities, strict=True
        ):
            # For deepinv.physics.PET, we need to add the background term
            if hasattr(cur_physics, "background"):
                proj = cur_physics.A(x, add_background=True)
            else:
                proj = cur_physics.A(x)

            numerator = x * cur_physics.A_adjoint(cur_y / proj.clamp(min=self.eps))
            denom = cur_sensitivity

            # Scale the OSL prior so that one full epoch applies its total weight.
            if cur_prior is not None:
                prior_grad = cur_prior.grad(x, cur_params["g_param"])
                denom = denom + prior_scale * cur_params["lambda"] * prior_grad

            x = numerator / denom.clamp(min=self.eps)

        F = None
        if self.has_cost and cur_prior is not None:
            F = self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)

        return {"est": (x, None), "cost": F, "it": k + 1}
