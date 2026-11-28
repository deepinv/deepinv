from collections.abc import Callable, Iterable
from typing import Literal, Sequence
import torch
from tqdm import tqdm
from deepinv.loss.metric import Metric
from deepinv.models import Reconstructor
from deepinv.physics import Physics
import warnings
import torch.nn as nn
from deepinv.optim_v2.backtracking import Backtracking
from deepinv.utils.decorators import _deprecated_alias
from deepinv.optim.prior import ZeroPrior, Prior
from deepinv.optim.data_fidelity import (
    DataFidelity,
    ZeroFidelity,
)
import functools


def no_grad_unless_unfold(forward):
    r"""Run ``forward`` under :func:`torch.no_grad` unless ``self.unfold`` is ``True``."""

    @functools.wraps(forward)
    def wrapper(self, *args, **kwargs):
        with torch.set_grad_enabled(self.unfold and torch.is_grad_enabled()):
            return forward(self, *args, **kwargs)

    return wrapper


def stack_history(values, batch_size):
    r"""
    Stack a history of per-iteration values into a tensor of shape ``(batch_size, len(values))`` on CPU.

    Each value is either a scalar, broadcast to the whole batch, or a tensor with one value per batch
    element.

    :param list values: one value per iteration.
    :param int batch_size: batch size :math:`B`.
    :return: (:class:`torch.Tensor`) history of shape ``(batch_size, len(values))``, on CPU.
    """
    if not values:
        return torch.empty(batch_size, 0)
    return torch.stack(
        [torch.as_tensor(v).flatten().expand(batch_size) for v in values], dim=1
    ).cpu()


def relative_residual(x_prev, x):
    r"""
    Relative residual :math:`\|x_k - x_{k-1}\| / \|x_k\|` for each element of the batch.

    :param torch.Tensor x_prev: previous iterate :math:`x_{k-1}`.
    :param torch.Tensor x: current iterate :math:`x_k`.
    :return: (:class:`torch.Tensor`) residual of shape ``(batch_size,)``.
    """
    return (x_prev - x).flatten(1).norm(dim=1) / (x.flatten(1).norm(dim=1) + 1e-06)


def get_iteration_param(param, iteration):
    r"""
    Return the value of an iteration-dependent parameter or module at a given iteration.

    :param float, torch.Tensor, Callable, torch.nn.Module param: parameter or schedule.
        A :class:`torch.nn.ModuleList` is indexed by ``iteration`` (or its single element is used at every
        iteration), other modules are returned as is, callables are evaluated at ``iteration``,
        tensors with at least one dimension are indexed by ``iteration``, and scalars are returned as is.
    :param int iteration: zero-based iteration index.
    :return: the parameter value at ``iteration``.
    """
    if isinstance(param, nn.ModuleList):
        return param[0] if len(param) == 1 else param[iteration]
    if isinstance(param, nn.Module):
        return param
    if callable(param):
        return param(iteration)
    if isinstance(param, torch.Tensor) and param.ndim > 0:
        return param[0] if len(param) == 1 else param[iteration]
    return param


def _is_constant(param):
    r"""
    Whether an iteration-dependent parameter takes the same value at every iteration.

    :param param: schedule, data-fidelity term or prior.
    :return: (bool) whether ``param`` is constant along the iterations.
    """
    if isinstance(param, nn.ModuleList):
        return len(param) == 1
    if isinstance(param, nn.Module):
        return True
    return (
        param is None
        or isinstance(param, (int, float))
        or (isinstance(param, torch.Tensor) and param.numel() == 1)
    )


def pgd_objective(x, y, physics, data_fidelity, prior, prior_weight, prior_param=None):
    r"""
    Objective :math:`F(x) = \datafid{x}{y} + \lambda \reg{x}` minimized by proximal gradient descent.

    :param torch.Tensor x: current iterate, with shape ``(batch_size, *image_shape)``.
    :param torch.Tensor y: measurements, with shape ``(batch_size, *measurement_shape)``.
    :param deepinv.physics.Physics physics: forward model.
    :param deepinv.optim.DataFidelity data_fidelity: data-fidelity term.
    :param deepinv.optim.Prior prior: prior term.
    :param float prior_weight: prior weight :math:`\lambda`.
    :param float prior_param: optional parameter to pass to the prior, e.g. noise level for a denoiser.
    :return: objective function value, with shape ``(batch_size,)``, or ``None`` if the prior is not explicit.
    """
    if not prior.explicit_prior:
        raise ValueError(
            f"The objective cannot be computed: the prior {type(prior).__name__} is not explicit "
            "(e.g. a denoiser), so its value g(x) is not available."
        )
    return data_fidelity(x, y, physics) + prior_weight * prior(x, prior_param)


class PGD(Reconstructor):
    r"""
    Proximal gradient descent for minimizing :math:`\datafid{x}{y} + \lambda \reg{x}` where :math:`\datafid{x}{y}` is the data-fidelity term, :math:`\reg{x}` is the regularization term.

    If the attribute ``first_step`` is set to ``"data_fidelity"`` (by default), the PGD iterations are given by

    .. math::

        x_{k+1} = \operatorname{prox}_{\gamma\lambda g}
        (x_k - \gamma\nabla f(x_k)).

    If the attribute ``first_step`` is set to ``"prior"``, the functions :math:`\datafid` and :math:`\reg` are inverted in the previous iteration.

    .. note::
        The data-fidelity terms and priors are registered as submodules (sequences are stored in a
        :class:`torch.nn.ModuleList`). Their parameters, e.g. the weights of a denoiser, are therefore moved
        with the model and trained along with it, independently of ``unfold`` and ``trainable_params``.
        To keep them fixed, freeze them before building the algorithm, e.g. ``prior.requires_grad_(False)``.
        Passing the same instance several times in a sequence shares its parameters across iterations.

    .. note::
        By default (``unfold=False``), the iterations run under :func:`torch.no_grad`: no computational graph
        is kept and the output does not require gradients, which saves memory at inference time.
        Set ``unfold=True`` to keep the gradients through the iterations, e.g. to train the algorithm
        parameters or a learned prior, or to differentiate the output with respect to the inputs.

    Backtracking can be used to adapt the stepsize at each iteration. Backtracking consists in choosing
    the largest stepsize :math:`\gamma` such that, at each iteration, sufficient decrease of the cost function :math:`F` is achieved.
    For example, for Armijo backtracking, set ``backtracking=Armijo()``. See :class:`deepinv.optim_v2.Armijo` for details.

    .. note::
        To use backtracking, the optimized function (i.e., both the data-fidelity and prior) must be explicit and provide a computable cost for the current iterate.
        If the prior is not explicit (e.g. a denoiser) i.e. its attribute ``has_objective`` is set to ``False``, backtracking is automatically disabled.

    .. note::
        The objective is only computed if the data-fidelity and the prior are explicit. It may change along the
        iterations if ``prior_weight`` or ``prior_param`` is given as a callable or as a sequence of several values,
        or if ``prior`` or ``data_fidelity`` is given as a sequence of several distinct instances (repeating the same
        instance, e.g. ``[prior] * max_iter``, is not considered a change). In this case, the objective of iteration
        :math:`k` is used at iteration :math:`k`, so that backtracking compares values of the same objective.

    :param deepinv.optim.DataFidelity | Sequence[deepinv.optim.DataFidelity] data_fidelity: data-fidelity term :math:`\datafid`,
        or a sequence of data-fidelity terms, one per iteration. A sequence must contain either a single element,
        used at every iteration, or at least ``max_iter`` elements. A sequence of distinct instances makes the objective
        iteration-dependent (see the notes above). If ``None``, the data-fidelity term is set to
        :class:`deepinv.optim.ZeroFidelity`, which is equivalent to no data-fidelity term. Default: ``None``.
    :param deepinv.optim.Prior | Sequence[deepinv.optim.Prior] prior: prior term :math:`\reg`,
        or a sequence of prior terms, one per iteration. A sequence must contain either a single element, used at every iteration,
        or at least ``max_iter`` elements. A sequence of distinct instances makes the objective iteration-dependent
        (see the notes above). If ``None``, the prior term is set to :class:`deepinv.optim.ZeroPrior`, which is equivalent to
        no prior term. Default: ``None``.
    :param str first_step: ``"data_fidelity"`` or ``"prior"``, indicating which term is used for the first gradient step.
        if ``"data_fidelity"``, the gradient step is on the data-fidelity term and the proximal step is on the prior.
        If ``"prior"``, the gradient step is on the prior and the proximal step is on the data-fidelity term.
        Default: ``"data_fidelity"``.
    :param float, torch.Tensor, Iterable, Callable stepsize: gradient stepsize
        :math:`\gamma`, as a constant, one value per iteration, or a callable
        receiving the zero-based iteration index. Default: ``1.0``.
    :param float, torch.Tensor, Iterable, Callable prior_weight: prior weight :math:`\lambda`, with the same forms as ``stepsize``.
        ``lambda_reg`` is accepted as a deprecated alias. Default: ``1.0``.
    :param float, torch.Tensor, Iterable, Callable prior_param: optional parameter to pass to the prior, e.g. noise level for a denoiser.
        ``g_param`` is accepted as a deprecated alias. Default: ``None``.
    :param float, torch.Tensor, Iterable, Callable sigma_denoiser: alias of ``prior_param`` for denoiser priors.
        Cannot be used together with ``prior_param``. Default: ``None``.
    :param int max_iter: number of iterations. Default: ``100``.
    :param bool show_progress_bar: display iteration progress. Default: ``False``.
    :param Sequence metrics: DeepInv metrics to evaluate after each iteration. By default, PSNR is computed. Metrics must have distinct class names. Default: ``None``.
    :param Backtracking | None backtracking: Backtracking algorithm to use for stepsize adaptation,
        or ``None`` for a fixed stepsize. Default: ``None``.
    :param bool unfold: whether to unfold the algorithm. If ``True``, the computational graph is kept through the
        iterations, so that gradients can be backpropagated through the algorithm, and the parameters listed in
        ``trainable_params`` are registered as learnable :class:`torch.nn.Parameter`. If ``False``, the iterations
        run under :func:`torch.no_grad` and the output does not require gradients. Default: ``False``.
    :param Sequence[str] trainable_params: when ``unfold`` is ``True``, this is the list of parameter names to register as learnable parameters, between ``"stepsize"``,``"prior_weight"`` and ``"prior_param"``.
        Default: ``["stepsize", "prior_weight", "prior_param"]``. For no trainable parameters, set to an empty list ``[]``.
        This does not affect the parameters of ``data_fidelity`` and ``prior``.
    :param bool early_stop: whether to stop the algorithm early if the solution has converged. Default: ``False``.
    :param str crit_conv: convergence criterion to be used for claiming convergence, either ``"residual"`` (residual
        of the iterate norm) or ``"cost"`` (on the cost function). Default: ``"residual"``.
    :param float thres_conv: convergence threshold for the chosen convergence criterion. Default: ``1e-5``.
    :param bool verbose: whether to print convergence information at each iteration. Default: ``False``.
    """

    @_deprecated_alias(lambda_reg="prior_weight", g_param="prior_param")
    def __init__(
        self,
        data_fidelity: DataFidelity | Sequence[DataFidelity] = None,
        prior: Prior | Sequence[Prior] = None,
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
        prior_param: (
            float
            | torch.Tensor
            | Iterable[float | torch.Tensor]
            | Callable[[int], float | torch.Tensor]
        ) = None,
        sigma_denoiser: (
            float
            | torch.Tensor
            | Iterable[float | torch.Tensor]
            | Callable[[int], float | torch.Tensor]
        ) = None,
        max_iter: int = 100,
        show_progress_bar: bool = False,
        metrics: Sequence[Metric] = None,
        backtracking: Backtracking | None = None,
        unfold: bool = False,
        trainable_params: Sequence[str] = ("stepsize", "prior_weight", "prior_param"),
        early_stop: bool = False,
        crit_conv: str = "residual",
        thres_conv: float = 1e-5,
        g_first: bool | None = None,  # backward compatibility
        verbose: bool = False,
    ):
        super().__init__()
        if g_first is not None:
            warnings.warn(
                "Argument 'g_first' is deprecated and will be removed in a future version. "
                "Use 'first_step' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            first_step = "prior" if g_first else "data_fidelity"
        if first_step not in ("data_fidelity", "prior"):
            raise ValueError(
                f"first_step must be 'data_fidelity' or 'prior', got {first_step!r}."
            )
        self.first_step = first_step
        self.max_iter = max_iter
        self.show_progress_bar = show_progress_bar
        if metrics is None:
            from deepinv.loss.metric import PSNR

            metrics = (PSNR(),)
        self.metrics = metrics
        # check that metrics have distinct class names
        names = [type(metric).__name__ for metric in metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"Metrics must have distinct class names, got {names}.")
        self.backtracking = backtracking
        self.unfold = unfold
        self._init_schedule(
            "stepsize", stepsize, trainable="stepsize" in trainable_params
        )
        self._init_schedule(
            "prior_weight", prior_weight, trainable="prior_weight" in trainable_params
        )
        if sigma_denoiser is not None:
            if prior_param is not None:
                raise TypeError(
                    "Cannot specify both 'prior_param' and 'sigma_denoiser'."
                )
            prior_param = sigma_denoiser
        self._init_schedule(
            "prior_param", prior_param, trainable="prior_param" in trainable_params
        )
        if data_fidelity is None:
            data_fidelity = ZeroFidelity()
        if prior is None:
            prior = ZeroPrior()
        self.data_fidelity = self._init_modules("data_fidelity", data_fidelity)
        self.prior = self._init_modules("prior", prior)
        priors = self.prior if isinstance(self.prior, nn.ModuleList) else [self.prior]
        self.has_objective = all(prior.explicit_prior for prior in priors)

        self.varying_objective = not all(
            _is_constant(param)
            for param in (
                self.prior_weight,
                self.prior_param,
                self.data_fidelity,
                self.prior,
            )
        )

        if not self.has_objective and self.backtracking is not None:
            warnings.warn("The prior is not explicit. Backtracking is disabled.")
            self.backtracking = None

        self.early_stop = early_stop
        self.crit_conv = crit_conv
        self.thres_conv = thres_conv
        if self.crit_conv not in ("residual", "cost"):
            raise ValueError(
                f"crit_conv must be 'residual' or 'cost', got {self.crit_conv!r}."
            )
        if self.crit_conv == "cost" and not self.has_objective:
            raise ValueError("crit_conv='cost' requires an explicit objective.")
        self.verbose = verbose

    def _init_schedule(self, name, param, trainable=False):
        r"""
        Store a stepsize or prior-weight iteration-dependent schedule as an attribute.

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
            and param is not None
            and not callable(param)
            and not isinstance(param, nn.Parameter)
        ):
            param = nn.Parameter(torch.as_tensor(param))
        setattr(self, name, param)

    def _init_modules(self, name, modules):
        r"""
        Store a data-fidelity term or prior, possibly changing at each iteration.

        Sequences are converted into a :class:`torch.nn.ModuleList` with one element per iteration,
        or a single element used at every iteration. Single modules are returned unchanged.
        """
        if not isinstance(modules, (list, tuple, nn.ModuleList)):
            return modules
        if 1 < len(modules) < self.max_iter or len(modules) == 0:
            raise ValueError(
                f"The number of elements in {name} must be 1 or at least max_iter."
            )
        return nn.ModuleList(modules)

    def _get_iteration_terms(self, iteration):
        r"""Return the stepsize, prior weight, prior parameter, data fidelity and prior used at ``iteration``."""
        return (
            get_iteration_param(self.stepsize, iteration),
            get_iteration_param(self.prior_weight, iteration),
            get_iteration_param(self.prior_param, iteration),
            get_iteration_param(self.data_fidelity, iteration),
            get_iteration_param(self.prior, iteration),
        )

    def _check_conv(self, residual, objective_prev, objective):
        r"""
        Check the convergence criterion.

        :param torch.Tensor residual: relative residual of the last step, shape ``(batch_size,)``.
        :param torch.Tensor objective_prev: objective at the previous iterate.
        :param torch.Tensor objective: objective at the current iterate.
        :return: (bool) ``True`` if the criterion is below ``thres_conv`` for all elements of the batch.
        """
        if self.crit_conv == "residual":
            crit = residual.max()
        else:
            crit = (
                (objective_prev - objective).abs() / (objective_prev.abs() + 1e-06)
            ).max()
        return bool(crit < self.thres_conv)

    def step(
        self,
        x,
        data_fidelity,
        prior,
        stepsize,
        prior_weight,
        prior_param,
        y,
        physics,
        grad=None,
    ):
        if self.first_step == "data_fidelity":
            if grad is None:
                grad = data_fidelity.grad(x, y, physics)
            return prior.prox(
                x - stepsize * grad,
                prior_param,
                gamma=stepsize * prior_weight,
            )
        else:
            if grad is None:
                grad = prior.grad(x, prior_param)
            return data_fidelity.prox(
                x - stepsize * prior_weight * grad, y, physics, gamma=stepsize
            )

    @no_grad_unless_unfold
    def forward(
        self,
        y: torch.Tensor,
        physics: Physics,
        init: torch.Tensor = None,
        x_gt: torch.Tensor = None,
        get_diagnostic: bool = False,
        get_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        r"""
        Run proximal gradient descent for minimizing :math:`\datafid{x}{y} + \lambda \reg{x}`.

        If ``unfold`` is ``False`` (default), this method runs under :func:`torch.no_grad`: the returned
        reconstruction does not require gradients. Set ``unfold=True`` at initialization to backpropagate
        through the algorithm.

        :param torch.Tensor y: measurements.
        :param deepinv.physics.Physics physics: forward model.
        :param torch.Tensor init: initial iterate. If ``None``, the initial iterate is set to :math:`x_0 = A^*(y)`. Default: ``None``.
        :param torch.Tensor x_gt: reference image for metrics. Default: ``None``.
        :param bool get_diagnostic: whether to return the metrics and parameters along iterations. Default: ``False``.
        :param bool get_trajectory: whether to store and return the trajectory in the diagnostic dictionary.
                Implies ``get_diagnostic=True``. Default: ``False``.
        :return: the reconstruction :math:`\hat{x}`. If ``get_diagnostic`` or ``get_trajectory`` is ``True``,
            returns ``(x, diagnostic)`` where ``diagnostic`` is a dictionary with keys:

            - ``"stepsize"``, ``"prior_weight"``: accepted values, shape ``(B, K)``,
            - ``"residual"``: :math:`\|x_k - x_{k-1}\| / \|x_k\|`, shape ``(B, K)``,
            - ``"objective"``: :math:`F(x_k)` for :math:`k = 0, \dots, K`, shape ``(B, K + 1)``, if the objective is
              explicit, where :math:`x_k` is evaluated with the prior weight, prior parameter and terms of the iteration
              that produced it.
            - ``"metrics"``: ``{name: values}`` of shape ``(B, K + 1)``, if ``x_gt`` is given,
            - ``"trajectory"``: list of iterates :math:`x_0, \dots, x_K` if ``get_trajectory`` is ``True``, else ``None``.

            :math:`K` is the number of iterations performed.
        """
        x = physics.A_adjoint(y) if init is None else init

        get_diagnostic = get_trajectory or get_diagnostic
        trajectory = [x.detach().clone()] if get_trajectory else None

        compute_objective = self.has_objective and (
            get_diagnostic
            or self.backtracking is not None
            or (self.early_stop and self.crit_conv == "cost")
        )

        if get_diagnostic:

            stepsize_values = []
            prior_weight_values = []
            prior_param_values = []
            objective_values = [] if self.has_objective else None
            residual_values = []
            if x_gt is None:
                if self.metrics:
                    warnings.warn(
                        "No ground truth provided. Metrics will not be computed.",
                    )
                metric_values = {}
            else:
                metric_values = {type(metric).__name__: [] for metric in self.metrics}

            # fill metrics values for the initial iterate
            with torch.no_grad():
                if x_gt is not None:
                    for metric in self.metrics:
                        metric_values[type(metric).__name__].append(
                            metric(x, x_gt).detach()
                        )

        objective = None

        for iteration in tqdm(range(self.max_iter), disable=not self.show_progress_bar):

            stepsize, prior_weight, prior_param, data_fidelity, prior = (
                self._get_iteration_terms(iteration)
            )

            if compute_objective and (iteration == 0 or self.varying_objective):
                with torch.no_grad():
                    objective = pgd_objective(
                        x, y, physics, data_fidelity, prior, prior_weight, prior_param
                    )
                    if get_diagnostic and iteration == 0:
                        objective_values.append(objective)

            x_prev, objective_prev = x, objective

            grad = (
                data_fidelity.grad(x, y, physics)
                if self.first_step == "data_fidelity"
                else prior.grad(x, prior_param)
            )

            num_trials = (
                self.backtracking.max_steps if self.backtracking is not None else 1
            )
            for _ in range(num_trials):
                x_next = self.step(
                    x,
                    data_fidelity,
                    prior,
                    stepsize,
                    prior_weight,
                    prior_param,
                    y,
                    physics,
                    grad,
                )
                if self.backtracking is None:
                    x = x_next
                    break
                with torch.no_grad():
                    objective_next = pgd_objective(
                        x_next,
                        y,
                        physics,
                        data_fidelity,
                        prior,
                        prior_weight,
                        prior_param,
                    )
                    accepted = self.backtracking.accept(
                        x,
                        x_next,
                        stepsize,
                        objective_x=objective_prev,
                        objective_candidate=objective_next,
                    )
                if accepted:
                    x = x_next
                    objective = objective_next
                    break
                stepsize = self.backtracking.update(stepsize)
                if self.verbose:
                    print(f"Backtracking: stepsize reduced to {float(stepsize):.3e}")

            else:  # no candidate accepted
                warnings.warn(
                    f"Backtracking failed after {num_trials} trials. Stopping at iteration {iteration}."
                )
                break

            if compute_objective and self.backtracking is None:
                with torch.no_grad():
                    objective = pgd_objective(
                        x, y, physics, data_fidelity, prior, prior_weight, prior_param
                    )

            residual = None
            if get_diagnostic or (self.early_stop and self.crit_conv == "residual"):
                with torch.no_grad():
                    residual = relative_residual(x_prev, x)

            if get_diagnostic:
                # Metric logging
                with torch.no_grad():
                    residual_values.append(residual)
                    if x_gt is not None:
                        for metric in self.metrics:
                            metric_values[type(metric).__name__].append(
                                metric(x, x_gt).detach()
                            )
                    if objective is not None:
                        objective_values.append(objective.detach())

                # Stepsize logging
                stepsize_values.append(torch.as_tensor(stepsize).detach().clone())
                prior_weight_values.append(
                    torch.as_tensor(prior_weight).detach().clone()
                )
                if prior_param is not None:
                    prior_param_values.append(
                        torch.as_tensor(prior_param).detach().clone()
                    )

            if get_trajectory:
                trajectory.append(x.detach().clone())

            if self.early_stop and self._check_conv(
                residual, objective_prev, objective
            ):
                if self.verbose:
                    print(
                        f"Iteration {iteration}, convergence criterion reached. Stopping early."
                    )
                break

        batch_size = x.shape[0]

        if not get_diagnostic:
            return x
        diagnostic = {
            "residual": stack_history(residual_values, batch_size),
            "metrics": {
                n: stack_history(v, batch_size) for n, v in metric_values.items()
            },
            "stepsize": stack_history(stepsize_values, batch_size),
            "prior_weight": stack_history(prior_weight_values, batch_size),
            "trajectory": trajectory,
        }
        if objective_values:
            diagnostic["objective"] = stack_history(objective_values, batch_size)
        if prior_param_values:
            diagnostic["prior_param"] = stack_history(prior_param_values, batch_size)
        return x, diagnostic
