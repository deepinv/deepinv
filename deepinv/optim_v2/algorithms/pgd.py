from collections.abc import Callable, Iterable
from typing import Literal, Sequence
import torch
from tqdm import tqdm
from deepinv.loss.metric import Metric
from deepinv.models import Reconstructor
from deepinv.optim import DataFidelity, Prior
from deepinv.physics import Physics
import warnings
import torch.nn as nn


class PGD(Reconstructor):
    r"""
    Proximal gradient descent for minimizing :math:`\datafid{x}{y} + \lambda \reg{x}` where :math:`\datafid{x}{y}` is the data-fidelity term, :math:`\reg{x}` is the regularization term.

    If the attribute ``first_step`` is set to ``"data_fidelity"`` (by default), the PGD iterations are given by

    .. math::

        x_{k+1} = \operatorname{prox}_{\gamma\lambda g}
        (x_k - \gamma\nabla f(x_k)).

    If the attribute ``first_step`` is set to ``"prior"``, the functions :math:`\datafid` and :math:`\reg` are inverted in the previous iteration.

    Backtracking can be used to adapt the stepsize at each iteration. Backtracking consists in choosing
    the largest stepsize :math:`\gamma` such that, at each iteration, sufficient decrease of the cost function :math:`F` is achieved.
    Armijo backtracking accepts a candidate when the mean objective decrease over the batch is at least :math:`\mathrm{decrease\_factor}\|x_{k+1}-x_k\|^2/\gamma`.
    If not accepted, the stepsize is reduced by a factor :math:`\mathrm{reduction\_factor}` and the candidate is recomputed.
    This is repeated for at most :math:`\mathrm{max\_backtracking\_steps}` times.

    .. note::
        To use backtracking, the optimized function (i.e., both the data-fidelity and prior) must be explicit and provide a computable cost for the current iterate.
        If the prior is not explicit (e.g. a denoiser) i.e. its attribute ``has_cost`` is set to ``False``, backtracking is automatically disabled.


    :param deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`f`.
    :param deepinv.optim.Prior prior: prior :math:`g`.
    :param str first_step: ``"data_fidelity"`` or ``"prior"``, indicating which term is used for the first gradient step.
        if ``"data_fidelity"``, the gradient step is on the data-fidelity term and the proximal step is on the prior.
        If ``"prior"``, the gradient step is on the prior and the proximal step is on the data-fidelity term.
        Default: ``"data_fidelity"``.
    :param float, torch.Tensor, Iterable, Callable stepsize: gradient stepsize
        :math:`\gamma`, as a constant, one value per iteration, or a callable
        receiving the zero-based iteration index. Default: ``1.0``.
    :param float, torch.Tensor, Iterable, Callable prior_weight: prior weight
        :math:`\lambda`, with the same forms as ``stepsize``. Default: ``1.0``.
    :param int max_iter: number of iterations. Default: ``100``.
    :param bool show_progress_bar: display iteration progress. Default: ``False``.
    :param Sequence metrics: DeepInv metrics to evaluate after each iteration. Default: ``()``.
    :param str backtracking: ``"armijo"`` for sufficient-decrease line search,
        or ``None`` for a fixed stepsize. Default: ``None``.
    :param float decrease_factor: Armijo-like parameter (controls sufficient decrease). Default: ``0.1``.
    :param float reduction_factor: Step reduction factor (e.g. multiply step by reduction_factor on failure). Default: ``0.5``.
    :param int max_backtracking_steps: Maximum number of backtracking steps. Default: ``10``.
    :param bool get_trajectory: whether to return the trajectory of iterates. Default: ``False``.
    :param bool unfold: whether to unfold the algorithm and make the model parameters trainable. Default: ``False``.
    :param Sequence[str] trainable_params: when ``unfold`` is ``True``, this is the list of parameter names to register as learnable parameters, between ``"stepsize"`` and ``"prior_weight"``.
        Default: ``["stepsize", "prior_weight"]``. For no trainable parameters, set to an empty list ``[]``.
    """

    def __init__(
        self,
        data_fidelity: DataFidelity,
        prior: Prior,
        first_step: Literal["data_fidelity", "prior"] = "data_fidelity",
        stepsize: (
            float
            | torch.Tensor
            | Iterable[float | torch.Tensor]
            | Callable[[int], float | torch.Tensor]
        ) = 1.0,
        prior_weight: (
            float
            | torch.Tensor
            | Iterable[float | torch.Tensor]
            | Callable[[int], float | torch.Tensor]
        ) = 1.0,
        max_iter: int = 100,
        show_progress_bar: bool = False,
        metrics: Sequence[Metric] = (),
        backtracking: Literal["armijo", None] = None,
        decrease_factor: float = 0.1,
        reduction_factor: float = 0.5,
        max_backtracking_steps: int = 10,
        get_trajectory: bool = False,
        unfold: bool = False,
        trainable_params: Sequence[str] = ("stepsize", "prior_weight"),
    ):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.first_step = first_step
        self.max_iter = max_iter
        self.show_progress_bar = show_progress_bar
        self.metrics = metrics
        self.backtracking = backtracking
        self.decrease_factor = decrease_factor
        self.reduction_factor = reduction_factor
        self.max_backtracking_steps = max_backtracking_steps
        self.get_trajectory = get_trajectory
        self.has_cost = self.prior.explicit_prior
        if not self.has_cost and self.backtracking is not None:
            warnings.warn(
                "The prior is not explicit (has_cost=False). Backtracking is disabled."
            )
            self.backtracking = None
        self.unfold = unfold
        self._init_schedule(
            "stepsize", stepsize, trainable="stepsize" in trainable_params
        )
        self._init_schedule(
            "prior_weight", prior_weight, trainable="prior_weight" in trainable_params
        )

    def _init_schedule(self, name, param, trainable=False):
        r"""
        Store a stepsize or prior-weight schedule as an attribute.

        Iterables (lists, tuples, generators) are converted once into a tensor with
        one value per iteration. If ``trainable`` is ``True``, non-callable values are wrapped
        in a :class:`torch.nn.Parameter` that keeps their shape.
        Callables and existing :class:`torch.nn.Parameter` objects are stored unchanged.

        :param str name: attribute name, e.g. ``"stepsize"``.
        :param float, torch.Tensor, Iterable, Callable param: schedule value.
        :param bool trainable: whether to register the value as a learnable parameter. Default: ``False``.
        """
        if isinstance(param, Iterable) and not isinstance(param, torch.Tensor):
            param = torch.stack([torch.as_tensor(p) for p in param])
        if (
            isinstance(param, torch.Tensor)
            and param.ndim > 0
            and 1 < len(param) < self.max_iter
        ):
            raise ValueError(
                f"The number of elements in the parameter {name} is inferior to max_iter."
            )
        if (
            trainable
            and self.unfold
            and not callable(param)
            and not isinstance(param, nn.Parameter)
        ):
            param = nn.Parameter(torch.as_tensor(param))
        setattr(self, name, param)

    def _get_param(self, param, iteration):
        r"""
        Return the value of a schedule at a given iteration.

        :param float, torch.Tensor, Callable param: schedule. Callables are evaluated
            at ``iteration``, tensors with at least one dimension are indexed by
            ``iteration``, and scalars are returned as is.
        :param int iteration: zero-based iteration index.
        :return: the parameter value at ``iteration``.
        """
        if callable(param):
            return param(iteration)
        if isinstance(param, torch.Tensor) and param.ndim > 0:
            return param[0] if len(param) == 1 else param[iteration]
        return param

    def step(self, x, stepsize, prior_weight, y, physics, grad=None):
        if self.first_step == "data_fidelity":
            if grad is None:
                grad = self.data_fidelity.grad(x, y, physics)
            return self.prior.prox(
                x - stepsize * grad,
                gamma=stepsize * prior_weight,
            )
        else:
            if grad is None:
                grad = self.prior.grad(x)
            return self.data_fidelity.prox(
                x - stepsize * prior_weight * grad,
                y,
                physics,
                gamma=stepsize,
            )

    def objective(self, x, y, physics, prior_weight):
        if self.has_cost:
            return self.data_fidelity(x, y, physics) + prior_weight * self.prior(x)
        else:
            return None

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
            ``"metrics"``, accepted ``"stepsize"``, and ``"prior_weight"``
            histories. Parameter histories contain detached tensors.
        """
        x = physics.A_adjoint(y) if init is None else init
        objective_values = [] if self.has_cost else None
        metric_values = {type(metric).__name__: [] for metric in self.metrics}
        stepsize_values = []
        prior_weight_values = []

        trajectory = [x.detach().clone()] if self.get_trajectory else None

        for iteration in tqdm(range(self.max_iter), disable=not self.show_progress_bar):

            stepsize = self._get_param(self.stepsize, iteration)
            prior_weight = self._get_param(self.prior_weight, iteration)

            grad = (
                self.data_fidelity.grad(x, y, physics)
                if self.first_step == "data_fidelity"
                else self.prior.grad(x)
            )

            # Backtracking loop
            if self.backtracking == "armijo":
                with torch.no_grad():
                    objective_prev = self.objective(x, y, physics, prior_weight)
                for _ in range(self.max_backtracking_steps):
                    candidate = self.step(x, stepsize, prior_weight, y, physics, grad)
                    with torch.no_grad():
                        objective = self.objective(candidate, y, physics, prior_weight)
                        squared_norm = (
                            (candidate - x).flatten(1).square().sum(dim=1).mean()
                        )
                        sufficient_decrease = (objective_prev - objective).mean() >= (
                            self.decrease_factor / stepsize
                        ) * squared_norm
                        if sufficient_decrease:
                            break
                    stepsize = stepsize * self.reduction_factor
                else:
                    raise RuntimeError("Armijo backtracking failed")
                x = candidate

            # Actual algorithm
            else:
                x = self.step(x, stepsize, prior_weight, y, physics, grad)
                if self.has_cost:
                    with torch.no_grad():
                        objective = self.objective(x, y, physics, prior_weight)

            # Objective logging
            if self.has_cost:
                objective_values.append(objective.detach())

            # Metric logging
            with torch.no_grad():
                if x_gt is not None:
                    for metric in self.metrics:
                        metric_values[type(metric).__name__].append(
                            metric(x, x_gt).detach()
                        )

            # Stepsize logging
            stepsize_values.append(torch.as_tensor(stepsize).detach().clone())
            prior_weight_values.append(torch.as_tensor(prior_weight).detach().clone())
            if self.get_trajectory:
                trajectory.append(x.detach().clone())

        return x, {
            "objective": objective_values,
            "metrics": metric_values,
            "stepsize": stepsize_values,
            "prior_weight": prior_weight_values,
            "trajectory": trajectory,
        }
