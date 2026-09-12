from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from deepinv.optim.data_fidelity import PoissonLikelihood

from .optim_iterator import OptimIterator

if TYPE_CHECKING:
    from deepinv.optim import StackedPhysicsDataFidelity, Prior
    from deepinv.physics import StackedLinearPhysics
    from deepinv.utils import TensorList


class BSREMIteration(OptimIterator):
    r"""
    Performs a single BSREM epoch, updating the estimate once per measurement subset.
    See :class:`deepinv.optim.BSREM` for algorithm details.

    :param float eps: Lower bound for division denominators and the reconstructed image. Default: ``1e-6``.
    :param float sensitivity_threshold: Sensitivity threshold defining the reconstruction support. Default: ``1e-2``.
    :param Callable cost_fn: Custom cost function evaluated after each epoch. Default: ``None``.
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
        cur_data_fidelity: StackedPhysicsDataFidelity,
        cur_prior: Prior,
        cur_params: dict,
        y: TensorList,
        physics: StackedLinearPhysics,
        sensitivities: list[torch.Tensor],
        *args,
        **kwargs,
    ) -> dict[str, tuple[torch.Tensor, None] | torch.Tensor | int | None]:
        r"""
        Perform one Block Sequential Regularized EM epoch.

        :param dict X: Dictionary containing the current iterate and estimated cost.
        :param deepinv.optim.StackedPhysicsDataFidelity cur_data_fidelity: Data-fidelity terms corresponding to the physics subsets.
        :param deepinv.optim.Prior cur_prior: Differentiable prior used for each subset update.
        :param dict cur_params: Algorithm parameters ``"stepsize"``, ``"lambda"``, and ``"g_param"``.
        :param deepinv.utils.TensorList y: Measurement subsets.
        :param deepinv.physics.StackedLinearPhysics physics: Physics operators corresponding to the measurement subsets.
        :param list[torch.Tensor] sensitivities: Precomputed sensitivity maps :math:`A_l^T\mathbf{1}` for each subset.
        :return: Dictionary ``{"est": (x, None), "cost": F, "it": k + 1}`` containing the updated iterate and estimated cost.
        """
        x = X["est"][0]
        k = 0 if "it" not in X else X["it"]
        num_subsets = len(physics)
        average_sensitivity = sum(sensitivities) / num_subsets
        preconditioner_denominator = average_sensitivity.clamp(min=self.eps)
        sensitivity_support = average_sensitivity > (
            self.sensitivity_threshold
            * average_sensitivity.amax(dim=tuple(range(2, x.ndim)), keepdim=True)
        )

        for cur_y, cur_physics, cur_sensitivity, data_fidelity in zip(
            y, physics, sensitivities, cur_data_fidelity.data_fidelity_list, strict=True
        ):
            gain = 1.0
            if isinstance(data_fidelity, PoissonLikelihood):
                gain = data_fidelity.gain
                projection = cur_physics.A(x) + gain * data_fidelity.bkg
                if not data_fidelity.d.denormalize:
                    cur_y = gain * cur_y
            elif hasattr(cur_physics, "background"):
                projection = cur_physics.A(x, add_background=True)
            else:
                projection = cur_physics.A(x)

            data_gradient = cur_sensitivity - cur_physics.A_adjoint(
                cur_y / projection.clamp(min=self.eps)
            )
            prior_gradient = (
                # The data update is gain times the count-domain gradient.
                gain
                * cur_params["lambda"]
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
