from typing import Sequence

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

    :param deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`f`.
    :param deepinv.optim.Prior prior: prior :math:`g`.
    :param float stepsize: gradient stepsize :math:`\gamma`. Default: ``1.0``.
    :param float prior_weight: prior weight :math:`\lambda`. Default: ``1.0``.
    :param int max_iter: number of iterations. Default: ``100``.
    :param bool show_progress_bar: display iteration progress. Default: ``False``.
    :param Sequence metrics: DeepInv metrics to evaluate after each iteration. Default: ``()``.
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
    ):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.prior_weight = prior_weight
        self.max_iter = max_iter
        self.show_progress_bar = show_progress_bar
        self.metrics = metrics

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
        :return: reconstruction and a dictionary with ``"objective"`` and
            ``"metrics"`` histories.
        """
        x = physics.A_adjoint(y) if init is None else init
        objective_values = []
        metric_values = {type(metric): [] for metric in self.metrics}
        for _ in tqdm(range(self.max_iter), disable=not self.show_progress_bar):
            x = x - self.stepsize * self.data_fidelity.grad(x, y, physics)
            x = self.prior.prox(x, gamma=self.prior_weight * self.stepsize)
            with torch.no_grad():
                objective = self.data_fidelity(
                    x, y, physics
                ) + self.prior_weight * self.prior(x)
                objective_values.append(objective.detach())
                for metric in self.metrics:
                    metric_values[type(metric)].append(metric(x, x_gt).detach())
        return x, {"objective": objective_values, "metrics": metric_values}
