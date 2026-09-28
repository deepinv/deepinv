from typing import Literal, Sequence

import torch
from tqdm import tqdm

from deepinv.loss.metric import Metric
from deepinv.models import Reconstructor
from deepinv.optim import DataFidelity, Prior
from deepinv.physics import Physics


class PGD(Reconstructor):
    r"""
    Proximal gradient descent for minimizing :math:`f(x) + \lambda g(x)`.

    Each iteration applies

    .. math::

        x_{k+1} = \operatorname{prox}_{\gamma\lambda g}
        (x_k - \gamma\nabla f(x_k)).

    Armijo backtracking accepts a candidate when the mean objective decrease
    is at least :math:`0.1\|x_{k+1}-x_k\|^2/\gamma`. It raises an error if no
    trial is accepted after ten retries.

    :param deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`f`.
    :param deepinv.optim.Prior prior: prior :math:`g`.
    :param float stepsize: gradient stepsize :math:`\gamma`. Default: ``1.0``.
    :param float prior_weight: prior weight :math:`\lambda`. Default: ``1.0``.
    :param int max_iter: number of iterations. Default: ``100``.
    :param bool show_progress_bar: display iteration progress. Default: ``False``.
    :param Sequence metrics: DeepInv metrics to evaluate after each iteration. Default: ``()``.
    :param str backtracking: ``"armijo"`` for sufficient-decrease line search,
        or ``None`` for a fixed stepsize. Armijo halves the stepsize for at most
        ten retries per iteration. Default: ``None``.
    """

    def __init__(
        self,
        data_fidelity: DataFidelity,
        prior: Prior,
        stepsize: float = 1.0,
        prior_weight: float = 1.0,
        max_iter: int = 100,
        show_progress_bar: bool = False,
        metrics: Sequence[Metric] = (),
        backtracking: Literal["armijo"] | None = None,
    ):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.prior_weight = prior_weight
        self.max_iter = max_iter
        self.show_progress_bar = show_progress_bar
        self.metrics = metrics
        self.backtracking = backtracking

    def forward(
        self,
        y: torch.Tensor,
        physics: Physics,
        init: torch.Tensor = None,
        x_gt: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        r"""
        Run proximal gradient descent from ``init`` or :math:`A^\top y`.

        :param torch.Tensor y: measurements.
        :param deepinv.physics.Physics physics: forward model.
        :param torch.Tensor init: initial iterate. Default: ``None``.
        :param torch.Tensor x_gt: reference image for metrics. Default: ``None``.
        :return: reconstruction and a dictionary with ``"objective"``,
            ``"metrics"``, and accepted ``"stepsize"`` histories.
        """
        x = physics.A_adjoint(y) if init is None else init
        objective_values = []
        metric_values = {type(metric): [] for metric in self.metrics}
        stepsize_values = []
        for _ in tqdm(range(self.max_iter), disable=not self.show_progress_bar):

            # Part of the actual algorithm
            grad = self.data_fidelity.grad(x, y, physics)
            stepsize = self.stepsize

            # Backtracking loop
            if self.backtracking == "armijo":
                with torch.no_grad():
                    objective_prev = self.data_fidelity(
                        x, y, physics
                    ) + self.prior_weight * self.prior(x)
                for _ in range(11):
                    candidate = self.prior.prox(
                        x - stepsize * grad, gamma=self.prior_weight * stepsize
                    )
                    with torch.no_grad():
                        objective = self.data_fidelity(
                            candidate, y, physics
                        ) + self.prior_weight * self.prior(candidate)
                        change = (candidate - x).reshape(candidate.shape[0], -1)
                        squared_norm = change.abs().square().sum(dim=1).mean()
                        sufficient_decrease = (
                            objective_prev - objective
                        ).mean() >= 0.1 / stepsize * squared_norm
                    if sufficient_decrease:
                        break
                    stepsize *= 0.5
                else:
                    raise RuntimeError("Armijo backtracking failed after 10 retries.")
                x = candidate

            # Actual algorithm
            else:
                x = self.prior.prox(
                    x - stepsize * grad, gamma=self.prior_weight * stepsize
                )

            # Objective logging
            with torch.no_grad():
                objective = self.data_fidelity(
                    x, y, physics
                ) + self.prior_weight * self.prior(x)
            objective_values.append(objective.detach())

            # Metric logging
            with torch.no_grad():
                for metric in self.metrics:
                    metric_values[type(metric)].append(metric(x, x_gt).detach())

            # Stepsize logging
            stepsize_values.append(stepsize)

        return x, {
            "objective": objective_values,
            "metrics": metric_values,
            "stepsize": stepsize_values,
        }
