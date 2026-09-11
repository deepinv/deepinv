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
    r"""Iterator for Block Sequential Regularized Expectation Maximization.

    One iteration performs a complete BSREM epoch, updating the estimate once
    for each measurement subset. The updates use the average sensitivity map
    as a diagonal preconditioner and apply the prior gradient with weight
    :math:`\lambda/L`, where :math:`L` is the number of subsets. Voxels outside
    the sensitivity support are fixed to ``eps``.

    See :class:`deepinv.optim.BSREM` for the update equations and references.

    :param float eps: positive value used to clamp projection and sensitivity
        denominators, and as the lower bound of the reconstructed image.
        Default: ``1e-6``.
    :param float sensitivity_threshold: relative threshold used to define the
        reconstruction support from the average sensitivity map. Default:
        ``1e-2``.
    :param Callable cost_fn: custom objective function evaluated after each
        epoch. Default: ``None``.
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
        r"""Perform one Block Sequential Regularized EM epoch.

        :param dict X: current algorithm state. ``X["est"][0]`` contains the
            image estimate and ``X["it"]`` contains the epoch index when
            available.
        :param deepinv.optim.StackedPhysicsDataFidelity cur_data_fidelity:
            data-fidelity terms corresponding to the physics subsets.
        :param deepinv.optim.Prior cur_prior: differentiable prior used to
            regularize each subset update.
        :param dict cur_params: current algorithm parameters. The iterator uses
            ``"stepsize"``, ``"lambda"``, and ``"g_param"``.
        :param deepinv.utils.TensorList y: measurement subsets.
        :param deepinv.physics.StackedLinearPhysics physics: forward operators
            corresponding to the measurement subsets.
        :param list[torch.Tensor] sensitivities: precomputed subset sensitivity
            maps :math:`A_l^T\mathbf{1}`.
        :return: updated algorithm state ``{"est": (x, None), "cost": F,
            "it": k + 1}``.
        :rtype: dict
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
