import sys
import warnings
from collections.abc import Iterable
import torch
from deepinv.optim.optim_iterators import *
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.prior import Zero
from deepinv.optim.data_fidelity import ZeroFidelity
from deepinv.loss.metric.distortion import PSNR
from deepinv.models import Reconstructor
from deepinv.optim.bregman import BregmanL2
import torch.nn as nn
from contextlib import nullcontext


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

    Setting ``unfold`` to ``True`` enables to turn this iterative optimization algorithm into an unfolded algorithm, i.e. an algorithm
    that can be trained end-to-end, with learnable parameters. These learnable parameters encompass the trainable parameters of the algorithm which
    can be chosen with the ``trainable_params`` argument
    (e.g. ``stepsize`` :math:`\gamma`, regularization parameter ``lambda_reg`` :math:`\lambda`, prior parameter (``g_param``) :math:`\sigma` ...)
    but also the trainable priors (e.g. a deep denoiser) or forward models.

    If ``DEQ`` is set to ``True``, the algorithm is unfolded as a Deep Equilibrium model, i.e. the algorithm is virtually unrolled infinitely leveraging the implicit function theorem.
    The backward pass is then performed using fixed point iterations to find solutions of the fixed-point equation

    .. math::

        \begin{equation}
        v = \left(\frac{\partial \operatorname{FixedPoint}(x^\star)}{\partial x^\star} \right )^{\top} v + u.
        \end{equation}

    where :math:`u` is the incoming gradient from the backward pass,
    and :math:`x^\star` is the equilibrium point of the forward pass. See `this tutorial <http://implicit-layers-tutorial.org/deep_equilibrium_models/>`_ for more details.

    .. note::

        For now DEQ is only possible with ProximalGradientDescent, HQS and GradientDescent optimization algorithms.


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
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior: regularization prior.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
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
    :param bool unfold: whether to unfold the algorithm and make the model parameters trainable. Default: ``False``.
    :param list trainable_params: list of the algorithmic parameters among the keys of the dictionery params_algo to be made trainable. Default: ``None``, which means that all parameters in params_algo are trainable. For no trainable parameters, set to an empty list ``[]``.
    :param bool DEQ: whether to use a Deep Equilibrium approach as unfolding strategy i.e. the  algorithm is virtually unrolled infinitely leveraging the implicit function theorem. Default: ``False``.
    :param bool DEQ_jacobian_free: whether to use a Jacobian-free approach for the backward pass in the Deep Equilibrium model. The expansive Jacobian is removed in the implicit differentiation theorem. See https://ojs.aaai.org/index.php/AAAI/article/view/20619. Default: ``False``.
    :param bool DEQ_anderson_acceleration_backward: whether to use Anderson acceleration for the backward pass in the Deep Equilibrium model. Default: ``False``.
    :param int DEQ_history_size_backward: size of the history of iterates used for Anderson acceleration in the backward pass in the Deep Equilibrium model. Default: ``5``.
    :param float DEQ_beta_anderson_acc_backward: momentum of the Anderson acceleration step in the backward pass in the Deep Equilibrium model. Default: ``1.0``.
    :param float DEQ_eps_anderson_acc_backward: regularization parameter of the Anderson acceleration step in the backward pass in the Deep Equilibrium model. Default: ``1e-4``.
    :param int DEQ_max_iter_backward: maximum number of iterations for the backward pass in the Deep Equilibrium model.
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
        unfold=False,
        DEQ=False,
        trainable_params=None,
        DEQ_jacobian_free=False,
        DEQ_max_iter_backward=50,
        DEQ_anderson_acceleration_backward=False,
        DEQ_history_size_backward=5,
        DEQ_beta_anderson_acc_backward=1.0,
        DEQ_eps_anderson_acc_backward=1e-4,
        verbose=False,
        device=torch.device("cpu"),
        **kwargs,
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
        self.unfold = unfold
        self.DEQ = DEQ
        self.DEQ_jacobian_free = DEQ_jacobian_free
        self.DEQ_max_iter_backward = DEQ_max_iter_backward
        self.DEQ_anderson_acceleration_backward = DEQ_anderson_acceleration_backward
        self.DEQ_history_size_backward = DEQ_history_size_backward
        self.DEQ_beta_anderson_acc_backward = DEQ_beta_anderson_acc_backward
        self.DEQ_eps_anderson_acc_backward = DEQ_eps_anderson_acc_backward
        self.device = device

        # By default, ``self.prior`` should be a list of elements of the class :meth:`deepinv.optim.Prior`. The user could want the prior to change at each iteration. If no prior is given, we set it to a zero prior.
        if prior is None:
            self.prior = [Zero()]
        elif not isinstance(prior, Iterable):
            self.prior = [prior]
        else:
            self.prior = prior

        # By default, ``self.data_fidelity`` should be a list of elements of the class :meth:`deepinv.optim.DataFidelity`. The user could want the data-fidelity to change at each iteration.
        if data_fidelity is None:
            self.data_fidelity = [ZeroFidelity()]
        elif not isinstance(data_fidelity, Iterable):
            self.data_fidelity = [data_fidelity]
        else:
            self.data_fidelity = data_fidelity

        self.has_cost = (
            self.prior[0].explicit_prior
            if isinstance(self.prior, list)
            else self.prior.explicit_prior
        )
        iterator.has_cost = self.has_cost

        # By default ``params_algo`` should contain a prior ``g_param`` parameter, set by default to ``None``.
        if "g_param" not in params_algo.keys():
            params_algo["g_param"] = None

        # Correct the 'lambda_reg' key to 'lambda' in params_algo if it exists.
        if "lambda_reg" in params_algo.keys():
            params_algo["lambda"] = params_algo.pop("lambda_reg")

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

        # set trainable parameters
        if self.unfold or self.DEQ:
            if trainable_params is not None and "lambda_reg" in trainable_params:
                trainable_params[trainable_params.index("lambda_reg")] = "lambda"
            if trainable_params is None:
                trainable_params = params_algo.keys()
            for param_key in trainable_params:
                if param_key in self.init_params_algo.keys():
                    param_value = self.init_params_algo[param_key]
                    self.init_params_algo[param_key] = nn.ParameterList(
                        [
                            (
                                nn.Parameter(torch.tensor(el).float().to(device))
                                if not isinstance(el, torch.Tensor)
                                else nn.Parameter(el.float().to(device))
                            )
                            for el in param_value
                        ]
                    )
            self.init_params_algo = nn.ParameterDict(self.init_params_algo)
            self.params_algo = self.init_params_algo.copy()
            # The prior (list of instances of :class:`deepinv.optim.Prior`), data_fidelity and bremgna_potentials are converted to a `nn.ModuleList` to be trainable.
            self.prior = nn.ModuleList(self.prior) if self.prior else None
            self.data_fidelity = (
                nn.ModuleList(self.data_fidelity) if self.data_fidelity else None
            )

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

    def DEQ_additional_step(self, X, y, physics, **kwargs):
        r"""
        For Deep Equilibrium models, performs an additional step at the equilibrium point
        to compute the gradient of the fixed point operator with respect to the input.

        :param dict X: dictionary defining the current update at the equilibrium point.
        :param torch.Tensor y: measurement vector.
        :param deepinv.physics.Physics physics: physics of the problem for the acquisition of ``y``.
        """

        # Once, at the equilibrium point, performs one additional iteration with gradient tracking.
        cur_data_fidelity = (
            self.update_data_fidelity_fn(self.max_iter - 1)
            if self.update_data_fidelity_fn
            else None
        )
        cur_prior = (
            self.update_prior_fn(self.max_iter - 1) if self.update_prior_fn else None
        )
        cur_params = (
            self.update_params_fn(self.max_iter - 1) if self.update_params_fn else None
        )
        x = self.fixed_point.iterator(
            X, cur_data_fidelity, cur_prior, cur_params, y, physics, **kwargs
        )["est"][0]

        if not self.DEQ_jacobian_free:
            # Another iteration for jacobian computation via automatic differentiation.
            x0 = x.clone().detach().requires_grad_()
            f0 = self.fixed_point.iterator(
                {"est": (x0,)},
                cur_data_fidelity,
                cur_prior,
                cur_params,
                y,
                physics,
                **kwargs,
            )["est"][0]

            # Add a backwards hook that takes the incoming backward gradient `X["est"][0]` and solves the fixed point equation
            def backward_hook(grad):
                class backward_iterator(OptimIterator):
                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)

                    def forward(self, X, *args, **kwargs):
                        return {
                            "est": (
                                torch.autograd.grad(
                                    f0, x0, X["est"][0], retain_graph=True
                                )[0]
                                + grad,
                            )
                        }

                # Use the :class:`deepinv.optim.fixed_point.FixedPoint` class to solve the fixed point equation
                def init_iterate_fn(y, physics, F_fn=None):
                    return {"est": (grad,)}  # initialize the fixed point algorithm.

                backward_FP = FixedPoint(
                    backward_iterator(),
                    init_iterate_fn=init_iterate_fn,
                    max_iter=self.DEQ_max_iter_backward,
                    check_conv_fn=self.check_conv_fn,
                    anderson_acceleration=self.DEQ_anderson_acceleration_backward,
                    history_size=self.DEQ_history_size_backward,
                    beta_anderson_acc=self.DEQ_beta_anderson_acc_backward,
                    eps_anderson_acc=self.DEQ_eps_anderson_acc_backward,
                )
                g = backward_FP({"est": (grad,)}, None)[0]["est"][0]
                return g

            if x.requires_grad:
                x.register_hook(backward_hook)

        return x

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
        train_context = (
            torch.no_grad() if not self.unfold or self.DEQ else nullcontext()
        )
        with train_context:
            X, metrics = self.fixed_point(
                y, physics, x_gt=x_gt, compute_metrics=compute_metrics, **kwargs
            )
        if self.DEQ:
            x = self.DEQ_additional_step(X, y, physics, **kwargs)
        else:
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
                        cur_params["lambda"].flatten(1, -1).to(prior_value.device)
                        * prior_value.flatten(1, -1)
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

    .. note::

        Since 0.3.1, instead of using this function, it is possible to define optimization algorithms using directly the algorithm name e.g.
        ``model = ProximalGradientDescent(data_fidelity, prior, ...)``.

    :param str, deepinv.optim.optim_iterators.OptimIterator iteration: either the name of the algorithm to be used,
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


class ADMM(BaseOptim):
    r"""
    ADMM module for solving the problem

    .. math::
        \begin{equation}
        \label{eq:min_prob}
        \tag{1}
        \underset{x}{\arg\min} \quad  \datafid{x}{y} + \lambda \reg{x},
        \end{equation}

    where :math:`\datafid{x}{y}` is the data-fidelity term, :math:`\reg{x}` is the regularization term.
    If the attribute ``g_first`` is set to False (by default), the ADMM iterations write (`see this paper <https://www.nowpublishers.com/article/Details/MAL-016>`_)

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\gamma f}(x_k - z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(u_{k+1} + z_k) \\
        z_{k+1} &= z_k + \beta (u_{k+1} - x_{k+1})
        \end{aligned}
        \end{equation*}

    where :math:`\gamma>0` is a stepsize and :math:`\beta>0` is a relaxation parameter.  If the attribute ``g_first`` is set to ``True``, the functions :math:`f` and :math:`\regname` are
    inverted in the previous iterations. The ADMM iterations are defined in the iterator class :class:`deepinv.optim.optim_iterators.ADMMIteration`.

    If the attribute ``unfold`` is set to ``True``, the algorithm is unfolded and the algorithmic parameters of the algorithm are trainable.
    By default, all the algorithm parameters are trainiable : the stepsize :math:`\gamma`, the regularization parameter :math:`\lambda`, the prior parameter and the relaxation parameter :math:`\beta`.
    Use the ``trainable_params`` argument to adjust the list of trainable parameters.

    :param list, deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`\datafid{x}{y}`.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior prior: regularization prior :math:`\reg{x}`.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
    :param float lambda_reg: regularization parameter :math:`\lambda`. Default: ``1.0``.
    :param float stepsize: stepsize parameter :math:`\gamma`. Default: ``1.0``.
    :param float beta: ADMM relaxation parameter :math:`\beta`. Default: ``1.0``.
    :param float g_param: parameter of the prior function. For example the noise level for a denoising prior. Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: ``100``.
    :param bool g_first: whether to perform the proximal step on :math:`\reg{x}` before that on :math:`\datafid{x}{y}`, or the opposite. Default: ``False``.
    :param bool unfold: whether to unfold the algorithm or not. Default: ``False``.
    :param list trainable_params: list of ADMM parameters to be trained if ``unfold`` is True. To choose between ``["lambda", "stepsize", "g_param", "beta"]``. Default: None, which means that all parameters are trainable if ``unfold`` is True. For no trainable parameters, set to an empty list.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param torch.device device: device to use for the algorithm. Default: ``torch.device("cpu")``.
    """

    def __init__(
        self,
        data_fidelity=None,
        prior=None,
        lambda_reg=1.0,
        stepsize=1.0,
        beta=1.0,
        g_param=None,
        max_iter=100,
        unfold=False,
        trainable_params=None,
        g_first=False,
        F_fn=None,
        device=torch.device("cpu"),
        **kwargs,
        # add an unfolded mode for DEQ
    ):
        params_algo = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "g_param": g_param,
            "beta": beta,
        }
        super(ADMM, self).__init__(
            ADMMIteration(g_first=g_first, F_fn=F_fn),
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
            device=device,
            **kwargs,
        )


class DRS(BaseOptim):
    r"""
    DRS module for solving the problem

    .. math::
        \begin{equation}
        \label{eq:min_prob}
        \tag{1}
        \underset{x}{\arg\min} \quad  \datafid{x}{y} + \lambda \reg{x},
        \end{equation}

    where :math:`\datafid{x}{y}` is the data-fidelity term, :math:`\reg{x}` is the regularization term.
     If the attribute ``g_first`` is set to False (by default), the DRS iterations are given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\gamma f}(z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(2*u_{k+1}-z_k) \\
        z_{k+1} &= z_k + \beta (x_{k+1} - u_{k+1})
        \end{aligned}
        \end{equation*}

    where :math:`\gamma>0` is a stepsize and :math:`\beta>0` is a relaxation parameter. If the attribute ``g_first`` is set to True, the functions :math:`f` and :math:`\regname` are inverted in the previous iteration.
    The DRS iterations are defined in the iterator class :class:`deepinv.optim.optim_iterators.DRSIteration`.

    If the attribute ``unfold`` is set to ``True``, the algorithm is unfolded and the parameters of the algorithm are trainable.
    By default, all the algorithm parameters are trainiable : the stepsize :math:`\gamma`, the regularization parameter :math:`\lambda`, the prior parameter and the relaxation parameter :math:`\beta`.
    Use the ``trainable_params`` argument to adjust the list of trainable parameters.

    :param list, deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`\datafid{x}{y}`.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior prior: regularization prior :math:`\reg{x}`.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
    :param float lambda_reg: regularization parameter :math:`\lambda`. Default: ``1.0``.
    :param float stepsize: stepsize parameter :math:`\gamma`. Default: ``1.0``.
    :param float beta: DRS relaxation parameter :math:`\beta`. Default: ``1.0``.
    :param float g_param: parameter of the prior function. For example the noise level for a denoising prior. Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: ``100``.
    :param bool g_first: whether to perform the proximal step on :math:`\reg{x}` before that on :math:`\datafid{x}{y}`, or the opposite. Default: ``False``.
    :param bool unfold: whether to unfold the algorithm or not. Default: ``False``.
    :param list trainable_params: list of DRS parameters to be trained if ``unfold`` is True. To choose between ``["lambda", "stepsize", "g_param", "beta"]``. Default: None, which means that all parameters are trainable if ``unfold`` is True. For no trainable parameters, set to an empty list.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param torch.device device: device to use for the algorithm. Default: ``torch.device("cpu")``.
    """

    def __init__(
        self,
        data_fidelity=None,
        prior=None,
        lambda_reg=1.0,
        stepsize=1.0,
        beta=1.0,
        g_param=None,
        max_iter=100,
        g_first=False,
        unfold=False,
        trainable_params=None,
        F_fn=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        params_algo = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "g_param": g_param,
            "beta": beta,
        }
        super(DRS, self).__init__(
            DRSIteration(g_first=g_first, F_fn=F_fn),
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
            device=device,
            **kwargs,
        )


class GradientDescent(BaseOptim):
    r"""
    Gradient Descent module for solving the problem

    .. math::
        \begin{equation}
        \label{eq:min_prob}
        \tag{1}
        \underset{x}{\arg\min} \quad  \datafid{x}{y} + \lambda \reg{x},
        \end{equation}

    where :math:`\datafid{x}{y}` is the data-fidelity term, :math:`\reg{x}` is the regularization term.

    The Gradient Descent iterations are given by

    .. math::
        \begin{equation*}
        x_{k+1} = x_k - \gamma \nabla f(x_k) - \gamma \lambda \nabla \regname(x_k)
        \end{equation*}

    where :math:`\gamma>0` is a stepsize. The Gradient Descent iterations are defined in the iterator class :class:`deepinv.optim.optim_iterators.GDIteration`.

    If the attribute ``unfold`` is set to ``True``, the algorithm is unfolded and the parameters of the algorithm are trainable.
    By default, all the algorithm parameters are trainiable : the stepsize :math:`\gamma`, the regularization parameter :math:`\lambda`, the prior parameter.
    Use the ``trainable_params`` argument to adjust the list of trainable parameters.

    :param list, deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`\datafid{x}{y}`.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior prior: regularization prior :math:`\reg{x}`.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
    :param float lambda_reg: regularization parameter :math:`\lambda`. Default: ``1.0``.
    :param float stepsize: stepsize parameter :math:`\gamma`. Default: ``1.0``.
    :param float g_param: parameter of the prior function. For example the noise level for a denoising prior. Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: ``100``.
    :param bool unfold: whether to unfold the algorithm or not. Default: ``False``.
    :param list trainable_params: list of GD parameters to be trained if ``unfold`` is True. To choose between ``["lambda", "stepsize", "g_param"]``. Default: None, which means that all parameters are trainable if ``unfold`` is True. For no trainable parameters, set to an empty list.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param torch.device device: device to use for the algorithm. Default: ``torch.device("cpu")``.
    """

    def __init__(
        self,
        data_fidelity=None,
        prior=None,
        lambda_reg=1.0,
        stepsize=1.0,
        g_param=None,
        max_iter=100,
        unfold=False,
        trainable_params=None,
        DEQ=False,
        DEQ_jacobian_free=False,
        DEQ_max_iter_backward=10,
        DEQ_anderson_acceleration_backward=False,
        DEQ_history_size_backward=5,
        DEQ_beta_anderson_acc_backward=0.5,
        DEQ_eps_anderson_acc_backward=1e-6,
        F_fn=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        params_algo = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "g_param": g_param,
        }
        super(GradientDescent, self).__init__(
            GDIteration(F_fn=F_fn),
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
            DEQ=DEQ,
            DEQ_jacobian_free=DEQ_jacobian_free,
            DEQ_max_iter_backward=DEQ_max_iter_backward,
            DEQ_anderson_acceleration_backward=DEQ_anderson_acceleration_backward,
            DEQ_history_size_backward=DEQ_history_size_backward,
            DEQ_beta_anderson_acc_backward=DEQ_beta_anderson_acc_backward,
            DEQ_eps_anderson_acc_backward=DEQ_eps_anderson_acc_backward,
            device=device,
            **kwargs,
        )


class HQS(BaseOptim):
    r"""
    Half-Quadratic Splitting module for solving the problem
    
    .. math::
        \begin{equation}
        \label{eq:min_prob}
        \tag{1}
        \underset{x}{\arg\min} \quad  \datafid{x}{y} + \lambda \reg{x},
        \end{equation}
    
    where :math:`\datafid{x}{y}` is the data-fidelity term, :math:`\reg{x}` is the regularization term.
    If the attribute ``g_first`` is set to False (by default), the HQS iterations are given by
    
    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= \operatorname{prox}_{\gamma f}(x_k) \\
        x_{k+1} &= \operatorname{prox}_{\sigma \lambda \regname}(u_k).
        \end{aligned}
        \end{equation*}
    
    If the attribute ``g_first`` is set to True, the functions :math:`f` and :math:`\regname` are inverted in the previous iteration.
    The HQS iterations are defined in the iterator class :class:`deepinv.optim.optim_iterators.HQSIteration`.
   
    If the attribute ``unfold`` is set to ``True``, the algorithm is unfolded and the parameters of the algorithm are trainable.
    By default, all the algorithm parameters are trainiable : the stepsize :math:`\gamma`, the regularization parameter :math:`\lambda`, the prior parameter.
    Use the ``trainable_params`` argument to adjust the list of trainable parameters.

    :param list, deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`\datafid{x}{y}`.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior prior: regularization prior :math:`\reg{x}`.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
    :param float lambda_reg: regularization parameter :math:`\lambda`. Default: ``1.0``.
    :param float stepsize: stepsize parameter :math:`\gamma`. Default: ``1.0``.
    :param float g_param: parameter of the prior function. For example the noise level for a denoising prior. Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: ``100``.
    :param bool g_first: whether to perform the proximal step on :math:`\reg{x}` before that on :math:`\datafid{x}{y}`, or the opposite. Default: ``False``.
    :param bool unfold: whether to unfold the algorithm or not. Default: ``False``.
    :param list trainable_params: list of HQS parameters to be trained if ``unfold`` is True. To choose between ``["lambda", "stepsize", "g_param"]``. Default: None, which means that all parameters are trainable if ``unfold`` is True. For no trainable parameters, set to an empty list.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param torch.device device: device to use for the algorithm. Default: ``torch.device("cpu")``.
    """

    def __init__(
        self,
        data_fidelity=None,
        prior=None,
        lambda_reg=1.0,
        stepsize=1.0,
        g_param=None,
        max_iter=100,
        g_first=False,
        unfold=False,
        trainable_params=None,
        DEQ=False,
        DEQ_jacobian_free=False,
        DEQ_max_iter_backward=10,
        DEQ_anderson_acceleration_backward=False,
        DEQ_history_size_backward=5,
        DEQ_beta_anderson_acc_backward=0.5,
        DEQ_eps_anderson_acc_backward=1e-6,
        F_fn=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        params_algo = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "g_param": g_param,
        }
        super(HQS, self).__init__(
            HQSIteration(g_first=g_first, F_fn=F_fn),
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
            DEQ=DEQ,
            DEQ_jacobian_free=DEQ_jacobian_free,
            DEQ_max_iter_backward=DEQ_max_iter_backward,
            DEQ_anderson_acceleration_backward=DEQ_anderson_acceleration_backward,
            DEQ_history_size_backward=DEQ_history_size_backward,
            DEQ_beta_anderson_acc_backward=DEQ_beta_anderson_acc_backward,
            DEQ_eps_anderson_acc_backward=DEQ_eps_anderson_acc_backward,
            device=device,
            **kwargs,
        )


class ProximalGradientDescent(BaseOptim):
    r"""
    Proximal Gradient Descent module for solving the problem

    .. math::
        \begin{equation}
        \label{eq:min_prob}
        \tag{1}
        \underset{x}{\arg\min} \quad  \datafid{x}{y} + \lambda \reg{x},
        \end{equation}

    where :math:`\datafid{x}{y}` is the data-fidelity term, :math:`\reg{x}` is the regularization term.
    If the attribute ``g_first`` is set to False (by default), the PGD iterations are given by

    .. math::
        \begin{equation*}
        x_{k+1} = \operatorname{prox}_{\gamma \lambda \regname}(x_k - \gamma \nabla f(x_k)).
        \end{equation*}

    If the attribute ``g_first`` is set to True, the functions :math:`f` and :math:`\regname` are inverted in the previous iteration.
    The PGD iterations are defined in the iterator class :class:`deepinv.optim.optim_iterators.PGDIteration`.

    If the attribute ``unfold`` is set to ``True``, the algorithm is unfolded and the parameters of the algorithm are trainable.
    By default, all the algorithm parameters are trainiable : the stepsize :math:`\gamma`, the regularization parameter :math:`\lambda`, the prior parameter.
    Use the ``trainable_params`` argument to adjust the list of trainable parameters.

    :param list, deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`\datafid{x}{y}`.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior prior: regularization prior :math:`\reg{x}`.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
    :param float lambda_reg: regularization parameter :math:`\lambda`. Default: ``1.0``.
    :param float stepsize: stepsize parameter :math:`\gamma`. Default: ``1.0``.
    :param float g_param: parameter of the prior function. For example the noise level for a denoising prior. Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: ``100``.
    :param bool g_first: whether to perform the proximal step on :math:`\reg{x}` before that on :math:`\datafid{x}{y}`, or the opposite. Default: ``False``.
    :param bool unfold: whether to unfold the algorithm or not. Default: ``False``.
    :param list trainable_params: list of PGD parameters to be trained if ``unfold`` is True. To choose between ``["lambda", "stepsize", "g_param"]``. Default: None, which means that all parameters are trainable if ``unfold`` is True. For no trainable parameters, set to an empty list.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param torch.device device: device to use for the algorithm. Default: ``torch.device("cpu")``.
    """

    def __init__(
        self,
        data_fidelity=None,
        prior=None,
        lambda_reg=1.0,
        stepsize=1.0,
        g_param=None,
        max_iter=100,
        g_first=False,
        unfold=False,
        trainable_params=None,
        DEQ=False,
        DEQ_jacobian_free=False,
        DEQ_max_iter_backward=10,
        DEQ_anderson_acceleration_backward=False,
        DEQ_history_size_backward=5,
        DEQ_beta_anderson_acc_backward=0.5,
        DEQ_eps_anderson_acc_backward=1e-6,
        F_fn=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        params_algo = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "g_param": g_param,
        }
        super(ProximalGradientDescent, self).__init__(
            PGDIteration(g_first=g_first, F_fn=F_fn),
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
            DEQ=DEQ,
            DEQ_jacobian_free=DEQ_jacobian_free,
            DEQ_max_iter_backward=DEQ_max_iter_backward,
            DEQ_anderson_acceleration_backward=DEQ_anderson_acceleration_backward,
            DEQ_history_size_backward=DEQ_history_size_backward,
            DEQ_beta_anderson_acc_backward=DEQ_beta_anderson_acc_backward,
            DEQ_eps_anderson_acc_backward=DEQ_eps_anderson_acc_backward,
            device=device,
            **kwargs,
        )


class FISTA(BaseOptim):
    r"""
    FISTA module for acceleration of the Proximal Gradient Descent algorithm.
    If the attribute ``g_first`` is set to False (by default), the FISTA iterations are given by
    
    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= z_k -  \gamma \nabla f(z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(u_k) \\
        z_{k+1} &= x_{k+1} + \alpha_k (x_{k+1} - x_k),
        \end{aligned}
        \end{equation*}
    
    where :math:`\gamma` is a stepsize that should satisfy :math:`\gamma \leq 1/\operatorname{Lip}(\|\nabla f\|)` and
    :math:`\alpha_k = (k+a-1)/(k+a)`,  with :math:`a` a parameter that should be strictly greater than 2.
    
    If the attribute ``g_first`` is set to True, the functions :math:`f` and :math:`\regname` are inverted in the previous iteration.
    The FISTA iterations are defined in the iterator class :class:`deepinv.optim.optim_iterators.FISTAIteration`.

    If the attribute ``unfold`` is set to ``True``, the algorithm is unfolded and the parameters of the algorithm are trainable.
    By default, all the algorithm parameters are trainiable : the stepsize :math:`\gamma`, the regularization parameter :math:`\lambda`, the prior parameter, and the parameter :math:`a` of the FISTA algorithm.
    Use the ``trainable_params`` argument to adjust the list of trainable parameters.

    :param list, deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`\datafid{x}{y}`.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior prior: regularization prior :math:`\reg{x}`.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
    :param float lambda_reg: regularization parameter :math:`\lambda`. Default: ``1.0``.
    :param float stepsize: stepsize parameter :math:`\gamma`. Default: ``1.0``.
    :param int a: parameter of the FISTA algorithm, should be strictly greater than 2. Default: ``3``.
    :param float g_param: parameter of the prior function. For example the noise level for a denoising prior. Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: ``100``.
    :param bool g_first: whether to perform the proximal step on :math:`\reg{x}` before that on :math:`\datafid{x}{y}`, or the opposite. Default: ``False``.
    :param bool unfold: whether to unfold the algorithm or not. Default: ``False``.
    :param list trainable_params: list of FISTA parameters to be trained if ``unfold`` is True. To choose between ``["lambda", "stepsize", "g_param", "a"]``. Default: None, which means that all parameters are trainable if ``unfold`` is True. For no trainable parameters, set to an empty list.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param torch.device device: device to use for the algorithm. Default: ``torch.device("cpu")``.
    """

    def __init__(
        self,
        data_fidelity=None,
        prior=None,
        lambda_reg=1.0,
        stepsize=1.0,
        a=3,
        g_param=None,
        max_iter=100,
        g_first=False,
        unfold=False,
        trainable_params=None,
        F_fn=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        params_algo = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "g_param": g_param,
            "a": a,
        }
        super(FISTA, self).__init__(
            FISTAIteration(g_first=g_first, F_fn=F_fn),
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
            device=device,
            **kwargs,
        )


class MirrorDescent(BaseOptim):
    r"""
    Mirror Descent or Bregman variant of the Gradient Descent algorithm. For a given convex potential :math:`h`, the iterations are given by
    
    .. math::
        \begin{equation*}
        \begin{aligned}
        v_{k} &= \nabla f(x_k) + \lambda \nabla g(x_k) \\
        x_{k+1} &= \nabla h^*(\nabla h(x_k) - \gamma v_{k})
        \end{aligned}
        \end{equation*}
    
    where :math:`\gamma>0` is a stepsize and :math:`h^*` is the convex conjugate of :math:`h`.
    The Mirror Descent iterations are defined in the iterator class :class:`deepinv.optim.optim_iterators.MDIteration`.

    If the attribute ``unfold`` is set to ``True``, the algorithm is unfolded and the parameters of the algorithm are trainable.
    By default, all the algorithm parameters are trainiable : the stepsize :math:`\gamma`, the regularization parameter :math:`\lambda`, the prior parameter.
    Use the ``trainable_params`` argument to adjust the list of trainable parameters.

    :param deepinv.optim.Bregman bregman_potential: Bregman potential used for Bregman optimization algorithms such as Mirror Descent. Default: ``BregmanL2()``.
    :param list, deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`\datafid{x}{y}`.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior prior: regularization prior :math:`\reg{x}`.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
    :param float lambda_reg: regularization parameter :math:`\lambda`. Default: ``1.0``.
    :param float stepsize: stepsize parameter :math:`\gamma`. Default: ``1.0``.
    :param float g_param: parameter of the prior function. For example the noise level for a denoising prior. Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: ``100``.
    :param bool unfold: whether to unfold the algorithm or not. Default: ``False``.
    :param list trainable_params: list of MD parameters to be trained if ``unfold`` is True. To choose between ``["lambda", "stepsize", "g_param"]``. Default: None, which means that all parameters are trainable if ``unfold`` is True. For no trainable parameters, set to an empty list.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param torch.device device: device to use for the algorithm. Default: ``torch.device("cpu")``.
    """

    def __init__(
        self,
        bregman_potential=BregmanL2(),
        data_fidelity=None,
        prior=None,
        lambda_reg=1.0,
        stepsize=1.0,
        g_param=None,
        max_iter=100,
        unfold=False,
        trainable_params=None,
        F_fn=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        params_algo = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "g_param": g_param,
        }
        super(MirrorDescent, self).__init__(
            MDIteration(F_fn=F_fn, bregman_potential=bregman_potential),
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
            device=device,
            **kwargs,
        )


class ProximalMirrorDescent(BaseOptim):
    r""" 
    Proximal Mirror Descent or Bregman variant of the Proximal Gradient Descent algorithm. For a given convex potential :math:`h`, the iterations are given by
    
    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= \nabla h^*(\nabla h(x_k) - \gamma \nabla f(x_k)) \\
        x_{k+1} &= \operatorname{prox^h}_{\gamma \lambda \regname}(u_k)
        \end{aligned}
        \end{equation*}
    
    where :math:`\gamma` is a stepsize that should satisfy :math:`\gamma \leq 2/L` with :math:`L` verifying :math:`Lh-f` is convex. 
    :math:`\operatorname{prox^h}_{\gamma \lambda \regname}` is the Bregman proximal operator, detailed in the method :meth:`deepinv.optim.Potential.bregman_prox`.
    The Proximal Mirror Descent iterations are defined in the iterator class :class:`deepinv.optim.optim_iterators.PMDIteration`.

    If the attribute ``unfold`` is set to ``True``, the algorithm is unfolded and the parameters of the algorithm are trainable.
    By default, all the algorithm parameters are trainiable : the stepsize :math:`\gamma`, the regularization parameter :math:`\lambda`, the prior parameter.
    Use the ``trainable_params`` argument to adjust the list of trainable parameters.
    
    :param deepinv.optim.Bregman bregman_potential: Bregman potential used for Bregman optimization algorithms such as Proximal Mirror Descent. Default: ``BregmanL2()``.
    :param list, deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`\datafid{x}{y}`.
          Either a single instance (same data-fidelity for each iteration) or a list of instances of
          :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior prior: regularization prior :math:`\reg{x}`.
          Either a single instance (same prior for each iteration) or a list of instances of
          :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
    :param float lambda_reg: regularization parameter :math:`\lambda`. Default: ``1.0``.
    :param float stepsize: stepsize parameter :math:`\gamma`. Default: ``1.0``.
    :param float g_param: parameter of the prior function. For example the noise level for a denoising prior. Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: ``100``.
    :param bool unfold: whether to unfold the algorithm or not. Default: ``False``.
    :param list trainable_params: list of PMD parameters to be trained if ``unfold`` is True. To choose between ``["lambda", "stepsize", "g_param"]``. Default: None, which means that all parameters are trainable if ``unfold`` is True. For no trainable parameters, set to an empty list.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param torch.device device: device to use for the algorithm. Default: ``torch.device("cpu")``.

    """

    def __init__(
        self,
        bregman_potential=BregmanL2(),
        data_fidelity=None,
        prior=None,
        lambda_reg=1.0,
        stepsize=1.0,
        g_param=None,
        max_iter=100,
        unfold=False,
        trainable_params=None,
        F_fn=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        params_algo = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "g_param": g_param,
        }
        super(ProximalMirrorDescent, self).__init__(
            PMDIteration(F_fn=F_fn, bregman_potential=bregman_potential),
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
            device=device,
            **kwargs,
        )


class PrimalDualCP(BaseOptim):
    r"""
    Class for a single iteration of the `Chambolle-Pock <https://hal.science/hal-00490826/document>`_ Primal-Dual (PD)
    algorithm for minimising :math:`F(Kx) + \lambda G(x)` or :math:`\lambda F(x) + G(Kx)` for generic functions :math:`F` and :math:`G`.
    Our implementation corresponds to Algorithm 1 of `<https://hal.science/hal-00490826/document>`_.

    If the attribute ``g_first`` is set to ``False`` (by default), the iteration is given by
    
    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\sigma F^*}(u_k + \sigma K z_k) \\
        x_{k+1} &= \operatorname{prox}_{\tau \lambda G}(x_k-\tau K^\top u_{k+1}) \\
        z_{k+1} &= x_{k+1} + \beta(x_{k+1}-x_k) \\
        \end{aligned}
        \end{equation*}
    
    where :math:`F^*` is the Fenchel-Legendre conjugate of :math:`F`, :math:`\beta>0` is a relaxation parameter, and :math:`\sigma` and :math:`\tau` are step-sizes that should
    satisfy :math:`\sigma \tau \|K\|^2 \leq 1`. 

    If the attribute ``g_first`` is set to ``True``, the functions :math:`F` and :math:`G` are inverted in the previous iteration.
    In particular, setting :math:`F = \distancename`, :math:`K = A` and :math:`G = \regname`, the above algorithms solves

    .. math::
        \begin{equation*}
        \underset{x}{\operatorname{min}} \,\,  \distancename(Ax, y) + \lambda \regname(x)
        \end{equation*}
    
    with a splitting on :math:`\distancename`.

    Note that the algorithm requires an intiliazation of the three variables :math:`x_0`, :math:`z_0` and :math:`u_0`.

    If the attribute ``unfold`` is set to ``True``, the algorithm is unfolded and the parameters of the algorithm are trainable.
    By default, the trainiable parameters are : the stepsize :math:`\sigma`, the stepsize :math:`\tau`, the regularization parameter :math:`\lambda`, the prior parameter and the relaxation parameter :math:`\beta`.
    Use the ``trainable_params`` argument to adjust the list of trainable parameters.

    The Proximal Dual CP iterations are defined in the iterator class :class:`deepinv.optim.optim_iterators.CPIteration`.

    :param Callable K: linear operator :math:`K` in the primal problem. Default: identity function.
    :param Callable K_adjoint: adjoint linear operator :math:`K^\top` in the primal problem. Default: identity function.
    :param list, deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`\datafid{x}{y}`.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :class:`deepinv.optim.DataFidelity` (distinct data fidelity for each iteration). Default: ``None`` corresponding to :math:`\datafid{x}{y} = 0`.
    :param list, deepinv.optim.Prior prior: regularization prior :math:`\reg{x}`.
        Either a single instance (same prior for each iteration) or a list of instances of
        :class:`deepinv.optim.Prior` (distinct prior for each iteration). Default: ``None`` corresponding to :math:`\reg{x} = 0`.
    :param float lambda_reg: regularization parameter :math:`\lambda`. Default: ``1.0``.
    :param float stepsize: stepsize parameter :math:`\tau`. Default: ``1.0``.
    :param float stepsize_dual: stepsize parameter :math:`\sigma`. Default: ``1.0``.
    :param float beta: PD relaxation parameter :math:`\beta`. Default: ``1.0``.
    :param float g_param: parameter of the prior function. For example the noise level for a denoising prior. Default: ``None``.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: ``100``.
    :param bool g_first: whether to perform the proximal step on :math:`\reg{x}` before that on :math:`\datafid{x}{y}`, or the opposite. Default: ``False``.
    :param bool unfold: whether to unfold the algorithm or not. Default: ``False``.
    :param list trainable_params: list of PD parameters to be trained if ``unfold`` is True. To choose between ``["lambda", "stepsize", "stepsize_dual", "g_param", "beta"]``. For no trainable parameters, set to an empty list.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param torch.device device: device to use for the algorithm. Default: ``torch.device("cpu")``.

    """

    def __init__(
        self,
        K=lambda x: x,
        K_adjoint=lambda x: x,
        data_fidelity=None,
        prior=None,
        lambda_reg=1.0,
        stepsize=1.0,
        stepsize_dual=1.0,
        beta=1.0,
        g_param=None,
        max_iter=100,
        unfold=False,
        trainable_params=["lambda", "stepsize", "stepsize_dual", "g_param", "beta"],
        g_first=False,
        F_fn=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        params_algo = {
            "lambda": lambda_reg,
            "stepsize": stepsize,
            "stepsize_dual": stepsize_dual,
            "g_param": g_param,
            "beta": beta,
            "K": K,
            "K_adjoint": K_adjoint,
        }
        super(PrimalDualCP, self).__init__(
            CPIteration(g_first=g_first, F_fn=F_fn),
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            max_iter=max_iter,
            unfold=unfold,
            trainable_params=trainable_params,
            device=device,
            **kwargs,
        )
