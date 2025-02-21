import sys
import warnings
from collections.abc import Iterable
import torch
from deepinv.optim.optim_iterators import *
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.prior import Zero
from deepinv.loss.metric.distortion import PSNR
from deepinv.models import Reconstructor


class BaseOptim(Reconstructor):
    r"""
    Class for optimization algorithms, consists in iterating a fixed-point operator.

    Module solving the problem

    .. math::
        \begin{equation}
        \label{eq:min_prob}
        \tag{1}
        \underset{x}{\arg\min} \quad  \datafid{x}{y} + \lambda \reg{x},
        \end{equation}


    where the first term :math:`\datafidname:\xset\times\yset \mapsto \mathbb{R}_{+}` enforces data-fidelity, the second
    term :math:`\regname:\xset\mapsto \mathbb{R}_{+}` acts as a regularization and
    :math:`\lambda > 0` is a regularization parameter. More precisely, the data-fidelity term penalizes the discrepancy
    between the data :math:`y` and the forward operator :math:`A` applied to the variable :math:`x`, as

    .. math::
        \datafid{x}{y} = \distance{Ax}{y}

    where :math:`\distance{\cdot}{\cdot}` is a distance function, and where :math:`A:\xset\mapsto \yset` is the forward
    operator (see :class:`deepinv.physics.Physics`)

    Optimization algorithms for minimising the problem above can be written as fixed point algorithms,
    i.e. for :math:`k=1,2,...`

    .. math::
        \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, g, A, y, ...)


    where :math:`x_k` is a variable converging to the solution of the minimization problem, and
    :math:`z_k` is an additional variable that may be required in the computation of the fixed point operator.

    The :func:`optim_builder` function can be used to instantiate this class with a specific fixed point operator.

    If the algorithm is minimizing an explicit and fixed cost function :math:`F(x) =  \datafid{x}{y} + \lambda \reg{x}`,
    the value of the cost function is computed along the iterations and can be used for convergence criterion.
    Moreover, backtracking can be used to adapt the stepsize at each iteration. Backtracking consists in choosing
    the largest stepsize :math:`\tau` such that, at each iteration, sufficient decrease of the cost function :math:`F` is achieved.
    More precisely, Given :math:`\gamma \in (0,1/2)` and :math:`\eta \in (0,1)` and an initial stepsize :math:`\tau > 0`,
    the following update rule is applied at each iteration :math:`k`:

    .. math::
        \text{ while } F(x_k) - F(x_{k+1}) < \frac{\gamma}{\tau} || x_{k-1} - x_k ||^2, \,\, \text{ do } \tau \leftarrow \eta \tau

    The variable ``params_algo`` is a dictionary containing all the relevant parameters for running the algorithm.
    If the value associated with the key is a float, the algorithm will use the same parameter across all iterations.
    If the value is list of length max_iter, the algorithm will use the corresponding parameter at each iteration.

    The variable ``data_fidelity`` is a list of instances of :class:`deepinv.optim.DataFidelity` (or a single instance).
    If a single instance, the same data-fidelity is used at each iteration. If a list, the data-fidelity can change at each iteration.
    The same holds for the variable ``prior`` which is a list of instances of :class:`deepinv.optim.Prior` (or a single instance).

    .. doctest::

        >>> import deepinv as dinv
        >>> # This minimal example shows how to use the BaseOptim class to solve the problem
        >>> #                min_x 0.5  ||Ax-y||_2^2 + \lambda ||x||_1
        >>> # with the PGD algorithm, where A is the identity operator, lambda = 1 and y = [2, 2].
        >>>
        >>> # Create the measurement operator A
        >>> A = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)
        >>> A_forward = lambda v: A @ v
        >>> A_adjoint = lambda v: A.transpose(0, 1) @ v
        >>>
        >>> # Define the physics model associated to this operator
        >>> physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
        >>>
        >>> # Define the measurement y
        >>> y = torch.tensor([2, 2], dtype=torch.float64)
        >>>
        >>> # Define the data fidelity term
        >>> data_fidelity = dinv.optim.data_fidelity.L2()
        >>>
        >>> # Define the prior
        >>> prior = dinv.optim.Prior(g = lambda x, *args: torch.norm(x, p=1))
        >>>
        >>> # Define the parameters of the algorithm
        >>> params_algo = {"stepsize": 0.5, "lambda": 1.0}
        >>>
        >>> # Define the fixed-point iterator
        >>> iterator = dinv.optim.optim_iterators.PGDIteration()
        >>>
        >>> # Define the optimization algorithm
        >>> optimalgo = dinv.optim.BaseOptim(iterator,
        ...                     data_fidelity=data_fidelity,
        ...                     params_algo=params_algo,
        ...                     prior=prior)
        >>>
        >>> # Run the optimization algorithm
        >>> with torch.no_grad(): xhat = optimalgo(y, physics)
        >>> print(xhat)
        tensor([1., 1.], dtype=torch.float64)


    :param deepinv.optim.OptimIterator iterator: Fixed-point iterator of the optimization algorithm of interest.
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
        e.g. the stepsize, regularisation parameter, denoising standard deviation.
        Each value of the dictionary can be either Iterable (distinct value for each iteration) or
        a single float (same value for each iteration).
        Default: ``{"stepsize": 1.0, "lambda": 1.0}``. See :any:`optim-params` for more details.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None``.
    :param list, deepinv.optim.Prior: regularization prior.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: 100.
    :param str crit_conv: convergence criterion to be used for claiming convergence, either ``"residual"`` (residual
        of the iterate norm) or ``"cost"`` (on the cost function). Default: ``"residual"``
    :param float thres_conv: value of the threshold for claiming convergence. Default: ``1e-05``.
    :param bool early_stop: whether to stop the algorithm once the convergence criterion is reached. Default: ``True``.
    :param bool has_cost: whether the algorithm has an explicit cost function or not. Default: `False`.
    :param dict custom_metrics: dictionary containing custom metrics to be computed at each iteration.
    :param bool backtracking: whether to apply a backtracking strategy for stepsize selection. Default: ``False``.
    :param float gamma_backtracking: :math:`\gamma` parameter in the backtracking selection. Default: ``0.1``.
    :param float eta_backtracking: :math:`\eta` parameter in the backtracking selection. Default: ``0.9``.
    :param Callable custom_init:  initializes the algorithm with ``custom_init(y, physics)``. If ``None`` (default value),
        the algorithm is initialized with the adjoint :math:`A^{\top}y` when the adjoint is defined,
        and with the observation `y` if the adjoint is not defined. Default: ``None``.
    :param Callable get_output: get the image output given the current dictionary update containing primal
        and auxiliary variables ``X = {('est' : (primal, aux)}``. Default : ``X['est'][0]``.
    :param bool anderson_acceleration: whether to use Anderson acceleration for accelerating the forward fixed-point iterations.
        Default: ``False``.
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
        get_output=lambda X: X["est"][0],
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
        self.get_output = get_output
        self.has_cost = has_cost

        # By default ``params_algo`` should contain a prior ``g_param`` parameter, set by default to ``None``.
        if "g_param" not in params_algo.keys():
            params_algo["g_param"] = None

        # By default ``params_algo`` should contain a regularization parameter ``lambda`` parameter, which multiplies the prior term ``g``. It is set by default to ``1``.
        if "lambda" not in params_algo.keys():
            params_algo["lambda"] = 1.0

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

        # By default, ``self.prior`` should be a list of elements of the class :class:`deepinv.optim.Prior`. The user could want the prior to change at each iteration. If no prior is given, we set it to a zero prior.
        if prior is None:
            self.prior = [Zero()]
        elif not isinstance(prior, Iterable):
            self.prior = [prior]
        else:
            self.prior = prior

        # By default, ``self.data_fidelity`` should be a list of elements of the class :class:`deepinv.optim.DataFidelity`. The user could want the data-fidelity to change at each iteration.
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
            verbose=verbose,
        )

    def update_params_fn(self, it):
        r"""
        For each parameter ``params_algo``, selects the parameter value for iteration ``it``
        (if this parameter depends on the iteration number).

        :param int it: iteration number.
        :return: a dictionary containing the parameters at iteration ``it``.
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
        :return: the prior at iteration ``it``.
        """
        cur_prior = self.prior[it] if len(self.prior) > 1 else self.prior[0]
        return cur_prior

    def update_data_fidelity_fn(self, it):
        r"""
        For each data_fidelity function in `data_fidelity`, selects the data_fidelity value for iteration ``it``
        (if this data_fidelity depends on the iteration number).

        :param int it: iteration number.
        :return: the data_fidelity at iteration ``it``.
        """
        cur_data_fidelity = (
            self.data_fidelity[it]
            if len(self.data_fidelity) > 1
            else self.data_fidelity[0]
        )
        return cur_data_fidelity

    def init_iterate_fn(self, y, physics, F_fn=None):
        r"""
        Initializes the iterate of the algorithm.
        The first iterate is stored in a dictionary of the form ``X = {'est': (x_0, u_0), 'cost': F_0}`` where:

            * ``est`` is a tuple containing the first primal and auxiliary iterates.
            * ``cost`` is the value of the cost function at the first iterate.

        By default, the first (primal, auxiliary) iterate of the algorithm is chosen as :math:`(A^{\top}y, A^{\top}y)`.
        A custom initialization is possible with the custom_init argument.

        :param torch.Tensor y: measurement vector.
        :param deepinv.physics: physics of the problem.
        :param F_fn: function that computes the cost function.
        :return: a dictionary containing the first iterate of the algorithm.
        """
        self.params_algo = (
            self.init_params_algo.copy()
        )  # reset parameters to initial values
        if self.custom_init:
            init_X = self.custom_init(y, physics)
        else:
            x_init, z_init = physics.A_adjoint(y), physics.A_adjoint(y)
            init_X = {"est": (x_init, z_init)}
        F = (
            F_fn(
                init_X["est"][0],
                self.update_data_fidelity_fn(0),
                self.update_prior_fn(0),
                self.update_params_fn(0),
                y,
                physics,
            )
            if self.has_cost and F_fn is not None
            else None
        )
        init_X["cost"] = F
        return init_X

    def init_metrics_fn(self, X_init, x_gt=None):
        r"""
        Initializes the metrics.

        Metrics are computed for each batch and for each iteration.
        They are represented by a list of list, and ``metrics[metric_name][i,j]`` contains the metric ``metric_name``
        computed for batch i, at iteration j.

        :param dict X_init: dictionary containing the primal and auxiliary initial iterates.
        :param torch.Tensor x_gt: ground truth image, required for PSNR computation. Default: ``None``.
        :return dict: A dictionary containing the metrics.
        """
        init = {}
        x_init = self.get_output(X_init)
        self.batch_size = x_init.shape[0]
        if x_gt is not None:
            psnr = [
                [PSNR()(x_init[i : i + 1], x_gt[i : i + 1]).cpu().item()]
                for i in range(self.batch_size)
            ]
        else:
            psnr = [[] for i in range(self.batch_size)]
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
        :param dict X_prev: dictionary containing the primal and dual previous iterates.
        :param dict X: dictionary containing the current primal and dual iterates.
        :param torch.Tensor x_gt: ground truth image, required for PSNR computation. Default: None.
        :return dict: a dictionary containing the updated metrics.
        """
        if metrics is not None:
            x_prev = self.get_output(X_prev)
            x = self.get_output(X)
            for i in range(self.batch_size):
                residual = (
                    ((x_prev[i] - x[i]).norm() / (x[i].norm() + 1e-06))
                    .detach()
                    .cpu()
                    .item()
                )
                metrics["residual"][i].append(residual)
                if x_gt is not None:
                    psnr = PSNR()(x[i : i + 1], x_gt[i : i + 1])
                    metrics["psnr"][i].append(psnr.cpu().item())
                if self.has_cost:
                    F = X["cost"][i]
                    metrics["cost"][i].append(F.detach().cpu().item())
                if self.custom_metrics is not None:
                    for custom_metric_name, custom_metric_fn in zip(
                        self.custom_metrics.keys(), self.custom_metrics.values()
                    ):
                        metrics[custom_metric_name][i].append(
                            custom_metric_fn(
                                metrics[custom_metric_name], x_prev[i], x[i]
                            )
                        )
        return metrics

    def check_iteration_fn(self, X_prev, X):
        r"""
        Performs stepsize backtracking.

        :param dict X_prev: dictionary containing the primal and dual previous iterates.
        :param dict X: dictionary containing the current primal and dual iterates.
        """
        if self.backtracking and self.has_cost and X_prev is not None:
            x_prev = X_prev["est"][0]
            x = X["est"][0]
            x_prev = x_prev.reshape((x_prev.shape[0], -1))
            x = x.reshape((x.shape[0], -1))
            F_prev, F = X_prev["cost"], X["cost"]
            diff_F, diff_x = (
                (F_prev - F).mean(),
                (torch.norm(x - x_prev, p=2, dim=-1) ** 2).mean(),
            )
            stepsize = self.params_algo["stepsize"][0]
            if diff_F < (self.gamma_backtracking / stepsize) * diff_x:
                check_iteration = False
                self.params_algo["stepsize"] = [self.eta_backtracking * stepsize]
                if self.verbose:
                    print(
                        f'Backtraking : new stepsize = {self.params_algo["stepsize"][0]:.6f}'
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
        :param dict X_prev: dictionary containing the primal and dual previous iterates.
        :param dict X: dictionary containing the current primal and dual iterates.
        :return bool: ``True`` if the algorithm has converged, ``False`` otherwise.
        """
        if self.crit_conv == "residual":
            x_prev = self.get_output(X_prev)
            x = self.get_output(X)
            x_prev = x_prev.reshape((x_prev.shape[0], -1))
            x = x.reshape((x.shape[0], -1))
            crit_cur = (
                (x_prev - x).norm(p=2, dim=-1) / (x.norm(p=2, dim=-1) + 1e-06)
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

    def forward(self, y, physics, x_gt=None, compute_metrics=False, **kwargs):
        r"""
        Runs the fixed-point iteration algorithm for solving :ref:`(1) <optim>`.

        :param torch.Tensor y: measurement vector.
        :param deepinv.physics.Physics physics: physics of the problem for the acquisition of ``y``.
        :param torch.Tensor x_gt: (optional) ground truth image, for plotting the PSNR across optim iterations.
        :param bool compute_metrics: whether to compute the metrics or not. Default: ``False``.
        :param kwargs: optional keyword arguments for the optimization iterator (see :class:`deepinv.optim.OptimIterator`)
        :return: If ``compute_metrics`` is ``False``,  returns (:class:`torch.Tensor`) the output of the algorithm.
                Else, returns (torch.Tensor, dict) the output of the algorithm and the metrics.
        """
        with torch.no_grad():
            X, metrics = self.fixed_point(
                y, physics, x_gt=x_gt, compute_metrics=compute_metrics, **kwargs
            )
            x = self.get_output(X)
            if compute_metrics:
                return x, metrics
            else:
                return x


def create_iterator(
    iteration, prior=None, F_fn=None, g_first=False, bregman_potential=None
):
    r"""
    Helper function for creating an iterator, instance of the :class:`deepinv.optim.OptimIterator` class,
    corresponding to the chosen minimization algorithm.

    :param str, deepinv.optim.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"PGD"`` (proximal gradient descent), ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford).
    :param list, deepinv.optim.Prior: regularization prior.
                            Either a single instance (same prior for each iteration) or a list of instances of
                            deepinv.optim.Prior (distinct prior for each iteration). Default: ``None``.
    :param Callable F_fn: Custom user input cost function. default: None.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. Default: False
    :param deepinv.optim.Bregman bregman_potential: Bregman potential used for Bregman optimization algorithms such as Mirror Descent. Default: ``None``, uses standart Euclidean optimization.
    """
    # If no prior is given, we set it to a zero prior.
    if prior is None:
        prior = Zero()
    # If no custom objective function F_fn is given but g is explicitly given, we have an explicit objective function.
    explicit_prior = (
        prior[0].explicit_prior if isinstance(prior, list) else prior.explicit_prior
    )
    if F_fn is None and explicit_prior:

        def F_fn(x, data_fidelity, prior, cur_params, y, physics):
            prior_value = prior(x, cur_params["g_param"], reduce=False)
            if prior_value.dim() == 0:
                reg_value = cur_params["lambda"] * prior_value
            else:
                if isinstance(cur_params["lambda"], float):
                    reg_value = (cur_params["lambda"] * prior_value).sum()
                else:
                    reg_value = (
                        cur_params["lambda"].flatten().to(prior_value.device)
                        * prior_value.flatten()
                    ).sum()
            return data_fidelity(x, y, physics) + reg_value

        has_cost = True  # boolean to indicate if there is a cost function to evaluate along the iterations
    else:
        has_cost = False
    # Create an instance of :class:`deepinv.optim.OptimIterator`.
    if isinstance(
        iteration, str
    ):  # If the name of the algorithm is given as a string, the correspondong class is automatically called.
        iterator_fn = str_to_class(iteration + "Iteration")
        return iterator_fn(
            g_first=g_first,
            F_fn=F_fn,
            has_cost=has_cost,
            bregman_potential=bregman_potential,
        )
    else:
        # If the iteration is directly given as an instance of OptimIterator, nothing to do
        return iteration


def optim_builder(
    iteration,
    max_iter=100,
    params_algo={"lambda": 1.0, "stepsize": 1.0, "g_param": 0.05},
    data_fidelity=None,
    prior=None,
    F_fn=None,
    g_first=False,
    bregman_potential=None,
    **kwargs,
):
    r"""
    Helper function for building an instance of the :class:`deepinv.optim.BaseOptim` class.

    :param str, deepinv.optim.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"GD"`` (gradient descent),
        ``"PGD"`` (proximal gradient descent), ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford).
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: 100.
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
                            e.g. the stepsize, regularisation parameter, denoising standart deviation.
                            Each value of the dictionary can be either Iterable (distinct value for each iteration) or
                            a single float (same value for each iteration). See :any:`optim-params` for more details.
                            Default: ``{"stepsize": 1.0, "lambda": 1.0}``.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
                            Either a single instance (same data-fidelity for each iteration) or a list of instances of
                            :class:`deepinv.optim.DataFidelity` (distinct data-fidelity for each iteration). Default: ``None``.
    :param list, deepinv.optim.Prior prior: regularization prior.
                            Either a single instance (same prior for each iteration) or a list of instances of
                            deepinv.optim.Prior (distinct prior for each iteration). Default: ``None``.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. Default: `False`
    :param deepinv.optim.Bregman bregman_potential: Bregman potential used for Bregman optimization algorithms such as Mirror Descent. Default: ``None``, uses standart Euclidean optimization.
    :param kwargs: additional arguments to be passed to the :class:`deepinv.optim.BaseOptim` class.
    :return: an instance of the :class:`deepinv.optim.BaseOptim` class.

    """
    iterator = create_iterator(
        iteration,
        prior=prior,
        F_fn=F_fn,
        g_first=g_first,
        bregman_potential=bregman_potential,
    )
    return BaseOptim(
        iterator,
        has_cost=iterator.has_cost,
        data_fidelity=data_fidelity,
        prior=prior,
        params_algo=params_algo,
        max_iter=max_iter,
        **kwargs,
    ).eval()


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
