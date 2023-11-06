import sys
import warnings
import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint
from collections.abc import Iterable
from deepinv.utils import cal_psnr
from deepinv.optim.optim_iterators import *


class BaseOptim(nn.Module):
    r"""
    Class for optimization algorithms, consists in iterating a fixed-point operator.

    Module solving the problem

    .. math::
        \begin{equation}
        \label{eq:min_prob}
        \tag{1}
        \underset{x}{\arg\min} \quad \lambda \datafid{x}{y} + \reg{x},
        \end{equation}


    where the first term :math:`\datafidname:\xset\times\yset \mapsto \mathbb{R}_{+}` enforces data-fidelity, the second
    term :math:`\regname:\xset\mapsto \mathbb{R}_{+}` acts as a regularization and
    :math:`\lambda > 0` is a regularization parameter. More precisely, the data-fidelity term penalizes the discrepancy
    between the data :math:`y` and the forward operator :math:`A` applied to the variable :math:`x`, as

    .. math::
        \datafid{x}{y} = \distance{Ax}{y}

    where :math:`\distance{\cdot}{\cdot}` is a distance function, and where :math:`A:\xset\mapsto \yset` is the forward
    operator (see :meth:`deepinv.physics.Physics`)

    Optimization algorithms for minimising the problem above can be written as fixed point algorithms,
    i.e. for :math:`k=1,2,...`

    .. math::
        \qquad x_{k+1} = \operatorname{FixedPoint}(x_k, f, g, A, y, ...)


    where :math:`x_k` is the current iterate i.e. the fixed-point varaible iterated by the algorithm.
    The fixed-point variable does not necessarily correspond to the minimizer of :math:`F`.

    The :func:`optim_builder` function can be used to instantiate this class with a specific fixed point operator.

    If the algorithm is minimizing an explicit and fixed cost function :math:`F(x) = \lambda \datafid{x}{y} + \reg{x}`,
    the value of the cost function is computed along the iterations and can be used for convergence criterion.
    Moreover, backtracking can be used to adapt the stepsize at each iteration. Backtracking consits in chosing
    the largest stepsize :math:`\tau` such that, at each iteration, sufficient decrease of the cost function :math:`F` is achieved.
    More precisely, Given :math:`\gamma \in (0,1/2)` and :math:`\eta \in (0,1)` and an initial stepsize :math:`\tau > 0`,
    the following update rule is applied at each iteration :math:`k`:

    .. math::
        \text{ while } F(x_k) - F(x_{k+1}) < \frac{\gamma}{\tau} || x_{k-1} - x_k ||^2 \text{ do } \tau \leftarrow \eta \tau

    The variable ``params_algo`` is a dictionary containing all the relevant parameters for running the algorithm.
    If the value associated with the key is a float, the algorithm will use the same parameter across all iterations.
    If the value is list of length max_iter, the algorithm will use the corresponding parameter at each iteration.

    The variable ``data_fidelity`` is a list of instances of :meth:`deepinv.optim.DataFidelity` (or a single instance).
    If a single instance, the same data-fidelity is used at each iteration. If a list, the data-fidelity can change at each iteration.
    The same holds for the variable ``prior`` which is a list of instances of :meth:`deepinv.optim.Prior` (or a single instance).

    ::

        # This minimal example shows how to use the BaseOptim class to solve the problem
        #                min_x 0.5 \lambda ||Ax-y||_2^2 + ||x||_1
        # with the PGD algorithm, where A is the identity operator, lambda = 1 and y = [2, 2].

        # Create the measurement operator A
        A = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)
        A_forward = lambda v: A @ v
        A_adjoint = lambda v: A.transpose(0, 1) @ v

        # Define the physics model associated to this operator
        physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

        # Define the measurement y
        y = torch.tensor([2, 2], dtype=torch.float64)

        # Define the data fidelity term
        data_fidelity = dinv.optim.data_fidelity.L2()

        # Define the prior
        prior = dinv.optim.Prior(g = lambda x, *args: torch.norm(x, p=1))

        # Define the parameters of the algorithm
        params_algo = {"stepsize": 0.5, "lambda": 1.0}

        # Define the fixed-point iterator
        iterator = dinv.optim.optim_iterators.PGDIteration()

        # Define the optimization algorithm
        optimalgo = dinv.optim.BaseOptim(iterator,
                            data_fidelity=data_fidelity,
                            params_algo=params_algo,
                            prior=prior)

        # Run the optimization algorithm
        xhat = optimalgo(y, physics)


    :param deepinv.optim.optim_iterators.OptimIterator iterator: Fixed-point iterator of the optimization algorithm of interest.
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
                            e.g. the stepsize, regularisation parameter, denoising standard deviation.
                            Each value of the dictionary can be either Iterable (distinct value for each iteration) or
                            a single float (same value for each iteration).
                            Default: `{"stepsize": 1.0, "lambda": 1.0}`. See :any:`optim-params` for more details.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
                            Either a single instance (same data-fidelity for each iteration) or a list of instances of
                            :meth:`deepinv.optim.DataFidelity` (distinct data-fidelity for each iteration). Default: `None`.
    :param list, deepinv.optim.Prior: regularization prior.
                            Either a single instance (same prior for each iteration) or a list of instances of
                            :meth:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: 100.
    :param str crit_conv: convergence criterion to be used for claiming convergence, either ``"residual"`` (residual
                          of the iterate norm) or `"cost"` (on the cost function). Default: ``"residual"``
    :param float thres_conv: value of the threshold for claiming convergence. Default: ``1e-05``.
    :param bool early_stop: whether to stop the algorithm once the convergence criterion is reached. Default: ``True``.
    :param bool has_cost: whether the algorithm has an explicit cost function or not. Default: `False`.
    :param dict custom_metrics: dictionary containing custom metrics to be computed at each iteration.
    :param bool backtracking: whether to apply a backtracking strategy for stepsize selection. Default: ``False``.
    :param float gamma_backtracking: :math:`\gamma` parameter in the backtracking selection. Default: ``0.1``.
    :param float eta_backtracking: :math:`\eta` parameter in the backtracking selection. Default: ``0.9``.
    :param function custom_init:  initializes the algorithm with ``custom_init(y, physics)``.
        If ``None`` (default value) algorithm is initilialized with :math:`A^Ty`. Default: ``None``.
    :param function custom_output: function to get a custom output from the iterated dictionary X. Default: ``None``.
    :param bool anderson_acceleration: whether to use Anderson acceleration for accelerating the forward fixed-point iterations. Default: ``False``.
    :param int history_size: size of the history of iterates used for Anderson acceleration. Default: ``5``.
    :param float beta_anderson_acc: momentum of the Anderson acceleration step. Default: ``1.0``.
    :param float eps_anderson_acc: regularization parameter of the Anderson acceleration step. Default: ``1e-4``.
    :param bool verbose: whether to print relevant information of the algorithm during its run,
        such as convergence criterion at each iterate. Default: ``False``.
    :return: a torch model that solves the optimization problem.
    """

    def __init__(
        self,
        iterator,
        params_algo={"lambda": 1.0, "stepsize": 1.0},
        data_fidelity=None,
        prior=None,
        max_iter=100,
        crit_conv="residual",
        thres_conv=1e-5,
        early_stop=False,
        has_cost=False,
        backtracking=False,
        gamma_backtracking=0.1,
        eta_backtracking=0.9,
        custom_metrics=None,
        custom_init=None,
        custom_output=None,
        anderson_acceleration=False,
        history_size=5,
        beta_anderson_acc=1.0,
        eps_anderson_acc=1e-4,
        verbose=False,
    ):
        super(BaseOptim, self).__init__()

        self.early_stop = early_stop
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.max_iter = max_iter
        self.backtracking = backtracking
        self.gamma_backtracking = gamma_backtracking
        self.eta_backtracking = eta_backtracking
        self.has_converged = False
        self.thres_conv = thres_conv
        self.custom_metrics = custom_metrics
        self.custom_init = custom_init
        self.custom_output = custom_output
        self.has_cost = has_cost

        # By default ``params_algo`` should contain a prior ``g_param`` parameter, set by default to ``None``.
        if "g_param" not in params_algo.keys():
            params_algo["g_param"] = None

        # By default ``params_algo`` should contain a relaxation ``beta`` parameter, set by default to 1..
        if "beta" not in params_algo.keys():
            params_algo["beta"] = 1.0

        # By default, each parameter in ``params_algo` is a list.
        # If given as a single number, we convert it to a list of 1 element.
        # If given as a list of more than 1 element, it should have lenght ``max_iter``.
        for key, value in zip(params_algo.keys(), params_algo.values()):
            if not isinstance(value, Iterable):
                params_algo[key] = [value]
            else:
                if len(params_algo[key]) > 1 and len(params_algo[key]) < self.max_iter:
                    raise ValueError(
                        f"The number of elements in the parameter {key} is inferior to max_iter."
                    )
        # If ``stepsize`` is a list of more than 1 element, backtracking is impossible.
        if (
            "stepsize" in params_algo.keys()
            and len(params_algo["stepsize"]) > 1
            and self.backtracking
        ):
            self.backtracking = False
            warnings.warn(
                "Backtracking impossible when stepsize is predefined as a list. Setting backtracking to False."
            )
        # If no cost function, backtracking is impossible.
        if not self.has_cost and self.backtracking:
            self.backtracking = False
            warnings.warn(
                "Backtracking impossible when no cost function is given. Setting backtracking to False."
            )

        # keep track of initial parameters in case they are changed during optimization (e.g. backtracking)
        self.init_params_algo = params_algo

        # By default, ``self.prior`` should be a list of elements of the class :meth:`deepinv.optim.Prior`. The user could want the prior to change at each iteration.
        if not isinstance(prior, Iterable):
            self.prior = [prior]
        else:
            self.prior = prior

        # By default, ``self.data_fidelity`` should be a list of elements of the class :meth:`deepinv.optim.DataFidelity`. The user could want the prior to change at each iteration.
        if not isinstance(data_fidelity, Iterable):
            self.data_fidelity = [data_fidelity]
        else:
            self.data_fidelity = data_fidelity

        # Initialize the fixed-point module
        self.fixed_point = FixedPoint(
            iterator=iterator,
            update_params_fn=self.update_params_fn,
            update_data_fidelity_fn=self.update_data_fidelity_fn,
            update_prior_fn=self.update_prior_fn,
            check_iteration_fn=self.check_iteration_fn,
            check_conv_fn=self.check_conv_fn,
            init_metrics_fn=self.init_metrics_fn,
            init_iterate_fn=self.init_iterate_fn,
            update_metrics_fn=self.update_metrics_fn,
            max_iter=max_iter,
            early_stop=early_stop,
            anderson_acceleration=anderson_acceleration,
            history_size=history_size,
            beta_anderson_acc=beta_anderson_acc,
            eps_anderson_acc=eps_anderson_acc,
        )

    def update_params_fn(self, it):
        r"""
        For each parameter ``params_algo``, selects the parameter value for iteration ``it``
        (if this parameter depends on the iteration number).

        :param int it: iteration number.
        :return: a dictionary containing the parameters of iteration ``it``.
        """
        cur_params_dict = {
            key: value[it] if len(value) > 1 else value[0]
            for key, value in zip(self.params_algo.keys(), self.params_algo.values())
        }
        return cur_params_dict

    def update_prior_fn(self, it):
        r"""
        For each prior function in `prior`, selects the prior value for iteration ``it``
        (if this prior depends on the iteration number).

        :param int it: iteration number.
        :return: a dictionary containing the prior of iteration ``it``.
        """
        cur_prior = self.prior[it] if len(self.prior) > 1 else self.prior[0]
        return cur_prior

    def update_data_fidelity_fn(self, it):
        r"""
        For each data_fidelity function in `data_fidelity`, selects the data_fidelity value for iteration ``it``
        (if this data_fidelity depends on the iteration number).

        :param int it: iteration number.
        :return: a dictionary containing the data_fidelity of iteration ``it``.
        """
        cur_data_fidelity = (
            self.data_fidelity[it]
            if len(self.data_fidelity) > 1
            else self.data_fidelity[0]
        )
        return cur_data_fidelity

    def init_iterate_fn(self, y, physics, cost_fn=None):
        r"""
        Initializes the iterate of the algorithm.
        The first iterate is stored in a dictionary with keys ``iterate`` , ``estimate`` and ``cost``, where:
            - ``iterate`` is the first fixed-point iterate of the algorithm. It has dimension NxBxCxHxW, where N is the number of images in the fixed-point variable (1 by default).
            - ``estimate`` is the first estimate of the algorithm. It has dimension BxCxHxW.
            - ``cost`` is the value of the cost function at the first estimate.
        The default initialization is defined in the iterator class (see :meth:`deepinv.optim.optim_iterators.OptimIterator.init_algo`).
        A different custom initialization is possible with the custom_init argument.

        :param torch.Tensor y: measurement vector.
        :param deepinv.physics: physics of the problem.
        :param cost_fn: function that computes the cost function.
        :return: a dictionary containing the first iterate of the algorithm.
        """
        self.params_algo = (
            self.init_params_algo.copy()
        )  # reset parameters to initial values
        if self.custom_init:
            init_X = self.custom_init(y, physics)
        else:
            init_X = self.fixed_point.iterator.init_algo(y, physics)

        cost = (
            cost_fn(
                init_X["estimate"],
                self.update_data_fidelity_fn(0),
                self.update_prior_fn(0),
                self.update_params_fn(0),
                y,
                physics,
            )
            if self.has_cost and cost_fn is not None
            else None
        )
        init_X["cost"] = cost
        return init_X

    def init_metrics_fn(self, X_init, x_gt=None):
        r"""
        Initializes the metrics.

        Metrics are computed for each batch and for each iteration.
        They are represented by a list of list, and ``metrics[metric_name][i,j]`` contains the metric ``metric_name``
        computed for batch i, at iteration j.

        :param dict X_init: dictionary containing the initial iterate, initial estimate and cost at the initial estimate.
        :param torch.Tensor x_gt: ground truth image, required for PSNR computation. Default: ``None``.
        :return dict: A dictionary containing the metrics.
        """
        est_init = X_init["estimate"]
        self.batch_size = est_init.shape[0]
        init = {}
        psnr = [[] for i in range(self.batch_size)]
        if x_gt is not None:
            out = self.custom_output(X_init) if self.custom_output else est_init
            for i in range(self.batch_size):
                psnr[i].append(cal_psnr(out[i], x_gt[i]))
        init["psnr"] = psnr
        if self.has_cost:
            init["cost"] = [[] for i in range(self.batch_size)]
        init["residual"] = [[] for i in range(self.batch_size)]
        if self.custom_metrics is not None:
            for custom_metric_name in self.custom_metrics.keys():
                init[custom_metric_name] = [[] for i in range(self.batch_size)]
        return init

    def update_metrics_fn(self, metrics, X_prev, X, x_gt=None):
        r"""
        Function that compute all the metrics, across all batches, for the current iteration.

        :param dict metrics: dictionary containing the metrics. Each metric is computed for each batch.
        :param dict X_prev: dictionary containing the previous iterate, previous estimate and cost at the previous estimate.
        :param dict X: dictionary containing the current iterate, current estimate and cost at the current estimate.
        :param torch.Tensor x_gt: ground truth image, required for PSNR computation. Default: None.
        :return dict: a dictionary containing the updated metrics.
        """
        if metrics is not None:
            est_prev = X_prev["estimate"]
            est = X["estimate"]
            for i in range(self.batch_size):
                residual = (
                    ((est_prev[i] - est[i]).norm() / (est[i].norm() + 1e-06))
                    .detach()
                    .cpu()
                    .item()
                )
                metrics["residual"][i].append(residual)
                if x_gt is not None:
                    # apply custom output function if given, for PSNR computation.
                    out = self.custom_output(X) if self.custom_output else est
                    psnr = cal_psnr(out[i], x_gt[i])
                    metrics["psnr"][i].append(psnr)
                if self.has_cost:
                    F = X["cost"][i]
                    metrics["cost"][i].append(F.detach().cpu().item())
                if self.custom_metrics is not None:
                    for custom_metric_name, custom_metric_fn in zip(
                        self.custom_metrics.keys(), self.custom_metrics.values()
                    ):
                        metrics[custom_metric_name][i].append(
                            custom_metric_fn(
                                metrics[custom_metric_name], est_prev[i], est[i]
                            )
                        )
        return metrics

    def check_iteration_fn(self, X_prev, X):
        r"""
        Performs stepsize backtracking.

        :param dict X_prev: dictionary containing the previous iterate, previous estimate and cost at the previous estimate.
        :param dict X: dictionary containing the current iterate, current estimate and cost at the current estimate.
        """
        if self.backtracking and self.has_cost and X_prev is not None:
            est_prev = X_prev["estimate"]
            est = X["estimate"]
            est_prev = est_prev.reshape((est_prev.shape[0], -1))
            est = est.reshape((est.shape[0], -1))
            F_prev, F = X_prev["cost"], X["cost"]
            diff_F, diff_x = (
                (F_prev - F).mean(),
                (torch.norm(est - est_prev, p=2, dim=-1) ** 2).mean(),
            )
            stepsize = self.params_algo["stepsize"][0]
            if diff_F < (self.gamma_backtracking / stepsize) * diff_x:
                check_iteration = False
                self.params_algo["stepsize"] = [self.eta_backtracking * stepsize]
                if self.verbose:
                    print(
                        f'Backtraking : new stepsize = {self.params_algo["stepsize"][0]:.3f}'
                    )
            else:
                check_iteration = True
            return check_iteration
        else:
            return True

    def check_conv_fn(self, it, X_prev, X):
        r"""
        Checks the convergence of the algorithm.

        :param int it: iteration number.
        :param dict X_prev: dictionary containing the previous iterate, previous estimate and cost at the previous estimate.
        :param dict X: dictionary containing the current iterate, current estimate and cost at the current estimate.
        :return bool: ``True`` if the algorithm has converged, ``False`` otherwise.
        """
        if self.crit_conv == "residual":
            batch_size = X["iterate"][0].shape[0]
            iterate_prev = torch.cat([el.view((batch_size, -1)) for el in X_prev["iterate"]], dim = 1)
            iterate = torch.cat([el.view((batch_size, -1)) for el in X["iterate"]], dim = 1)
            crit_cur = (
                (iterate_prev - iterate).norm(p=2, dim=-1) / (iterate.norm(p=2, dim=-1) + 1e-06)
            ).mean()
        elif self.crit_conv == "cost":
            F_prev = X_prev["cost"]
            F = X["cost"]
            crit_cur = ((F_prev - F).norm(dim=-1) / (F.norm(dim=-1) + 1e-06)).mean()
        else:
            raise ValueError("convergence criteria not implemented")
        if crit_cur < self.thres_conv:
            self.has_converged = True
            if self.verbose:
                print(
                    f"Iteration {it}, current converge crit. = {crit_cur:.2E}, objective = {self.thres_conv:.2E} \r"
                )
            return True
        else:
            return False

    def forward(self, y, physics, x_gt=None, compute_metrics=False):
        r"""
        Runs the fixed-point iteration algorithm for solving :ref:`(1) <optim>`.

        :param torch.Tensor y: measurement vector.
        :param deepinv.physics physics: physics of the problem for the acquisition of ``y``.
        :param torch.Tensor x_gt: (optional) ground truth image, for plotting the PSNR across optim iterations.
        :param bool compute_metrics: whether to compute the metrics or not. Default: ``False``.
        :return: If ``compute_metrics`` is ``False``,  returns (torch.Tensor) the output of the algorithm.
                Else, returns (torch.Tensor, dict) the output of the algorithm and the metrics.
        """
        X, metrics = self.fixed_point(
            y, physics, x_gt=x_gt, compute_metrics=compute_metrics
        )
        out = self.custom_output(X) if self.custom_output else X["estimate"]
        if compute_metrics:
            return out, metrics
        else:
            return out


def create_iterator(iteration, prior=None, cost_fn=None, g_first=False):
    r"""
    Helper function for creating an iterator, instance of the :meth:`deepinv.optim.optim_iterators.OptimIterator` class,
    corresponding to the chosen minimization algorithm.

    :param str, deepinv.optim.optim_iterators.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"PGD"`` (proximal gradient descent), ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford).
    :param list, deepinv.optim.Prior: regularization prior.
                            Either a single instance (same prior for each iteration) or a list of instances of
                            deepinv.optim.Prior (distinct prior for each iteration). Default: `None`.
    :param callable cost_fn: Custom user input cost function. default: None.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. Default: False
    """
    # If no custom objective function cost_fn is given but g is explicitly given, we have an explicit objective function.
    explicit_prior = (
        prior[0].explicit_prior if isinstance(prior, list) else prior.explicit_prior
    )
    if cost_fn is None and explicit_prior:

        def cost_fn(x, data_fidelity, prior, cur_params, y, physics):
            return cur_params["lambda"] * data_fidelity(x, y, physics) + prior(
                x, cur_params["g_param"]
            )

        has_cost = True  # boolean to indicate if there is a cost function to evaluate along the iterations
    else:
        has_cost = False
    # Create an instance of :class:`deepinv.optim.optim_iterators.OptimIterator`.
    if isinstance(
        iteration, str
    ):  # If the name of the algorithm is given as a string, the correspondong class is automatically called.
        iterator_fn = str_to_class(iteration + "Iteration")
        return iterator_fn(g_first=g_first, cost_fn=cost_fn, has_cost=has_cost)
    else:
        # If the iteration is directly given as an instance of OptimIterator, nothing to do
        return iteration


def optim_builder(
    iteration,
    params_algo={"lambda": 1.0, "stepsize": 1.0},
    data_fidelity=None,
    prior=None,
    cost_fn=None,
    g_first=False,
    **kwargs,
):
    r"""
    Helper function for building an instance of the :meth:`BaseOptim` class.

    :param str, deepinv.optim.optim_iterators.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"PGD"`` (proximal gradient descent), ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford).
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
                            e.g. the stepsize, regularisation parameter, denoising standart deviation.
                            Each value of the dictionary can be either Iterable (distinct value for each iteration) or
                            a single float (same value for each iteration). See :any:`optim-params` for more details.
                            Default: ``{"stepsize": 1.0, "lambda": 1.0}``.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
                            Either a single instance (same data-fidelity for each iteration) or a list of instances of
                            :meth:`deepinv.optim.DataFidelity` (distinct data-fidelity for each iteration). Default: `None`.
    :param list, deepinv.optim.Prior prior: regularization prior.
                            Either a single instance (same prior for each iteration) or a list of instances of
                            deepinv.optim.Prior (distinct prior for each iteration). Default: `None`.
    :param callable cost_fn: Custom user input cost function. default: `None`.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: `False`
    :param kwargs: additional arguments to be passed to the :meth:`BaseOptim` class.
    :return: an instance of the :meth:`BaseOptim` class.

    """
    iterator = create_iterator(iteration, prior=prior, cost_fn=cost_fn, g_first=g_first)
    return BaseOptim(
        iterator,
        has_cost=iterator.has_cost,
        data_fidelity=data_fidelity,
        prior=prior,
        params_algo=params_algo,
        **kwargs,
    )


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
