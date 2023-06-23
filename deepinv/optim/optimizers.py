import sys

import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.data_fidelity import L2
from collections.abc import Iterable
from deepinv.utils import cal_psnr
from deepinv.optim.optim_iterators import *


class BaseOptim(nn.Module):
    r"""
    Class for optimization algorithms iterating the fixed-point iterator.

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
        \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, g, A, y, ...)


    where :math:`x_k` is a variable converging to the solution of the minimisation problem, and
    :math:`z_k` is an additional variable that may be required in the computation of the fixed point operator.


    ::

        # This example shows how to use the FixedPoint class to solve the problem
        #                min_x 0.5*lambda*||Ax-y||_2^2 + ||x||_1
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
        data_fidelity = L2()

        # Define the proximity operator of the prior and store it in a dictionary
        def prox_g(x, g_param=0.1):
            return torch.sign(x) * torch.maximum(x.abs() - g_param, torch.tensor([0]))

        prior = {"prox_g": prox_g}

        # Define the parameters of the algorithm
        params_algo = {"g_param": 0.5, "stepsize": 0.5, "lambda": 1.0}

        # Define the optimization algorithm
        iterator = PGDIteration(data_fidelity=data_fidelity)
        optimalgo = BaseOptim(iterator, prior=prior, params_algo=params_algo)

        # Run the optimization algorithm
        sol = optimalgo(y, physics)



    :param deepinv.optim.iterator iterator: Fixed-point iterator of the class of the algorithm of interest.
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
                             e.g. the stepsize, regularisation parameters, denoising power...
    :param dict prior: dictionary containing the regularization prior under the form of a denoiser, proximity operator,
                       gradient, or simply an auto-differentiable function. Default: {}
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: 50.
    :param str crit_conv: convergence criterion to be used for claiming convergence, either `"residual"` (residual
                          of the iterate norm) or `"cost"` (on the cost function). Default: `"residual"`
    :param float thres_conv: value of the threshold for claiming convergence. Default: `1e-05`.
    :param bool early_stop: whether to stop the algorithm once the convergence criterion is reached. Default: `True`.
    :param bool has_cost: whether the algorithm has a cost function or not. Default: `False`.
    :param bool return_aux: whether to return the auxiliary variable or not at the end of the algorithm. Default: `False`.
    :param bool backtracking: whether to apply a backtracking for stepsize selection. Default: `False`.
    :param float gamma_backtracking: :math:`\gamma` parameter in the backtracking selection. Default: `0.1`.
    :param float eta_backtracking: :math:`\eta` parameter in the backtracking selection. Default: `0.9`.
    :param function custom_init:  intializes the algorithm with `custom_init(y)`. If `None` (default value) algorithm is initilialized with :math:`A^Ty`. Default: `None`.
    :param bool verbose: whether to print relevant information of the algorithm during its run,
                         such as convergence criterion at each iterate. Default: `False`.
    """

    def __init__(
        self,
        iterator,
        params_algo={"lambda": 1.0, "stepsize": 1.0},
        prior={},
        max_iter=50,
        crit_conv="residual",
        thres_conv=1e-5,
        early_stop=False,
        has_cost=False,
        return_aux=False,
        backtracking=False,
        gamma_backtracking=0.1,
        eta_backtracking=0.9,
        return_metrics=False,
        custom_metrics=None,
        custom_init=None,
        verbose=False,
    ):
        super(BaseOptim, self).__init__()

        self.early_stop = early_stop
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.max_iter = max_iter
        self.return_aux = return_aux
        self.backtracking = backtracking
        self.gamma_backtracking = gamma_backtracking
        self.eta_backtracking = eta_backtracking
        self.return_metrics = return_metrics
        self.has_converged = False
        self.thres_conv = thres_conv
        self.custom_metrics = custom_metrics
        self.custom_init = custom_init
        self.has_cost = has_cost

        # params_algo should contain a g_param parameter, even if None.
        if "g_param" not in params_algo.keys():
            params_algo["g_param"] = None

        # By default, each parameter in params_algo is a list.
        # If given as a signel number, we convert it to a list of 1 element.
        # If given as a list of more than 1 element, it should have lenght max_iter.
        for key, value in zip(params_algo.keys(), params_algo.values()):
            if not isinstance(value, Iterable):
                params_algo[key] = [value]
            else:
                if len(params_algo[key]) > 1 and len(params_algo[key]) < self.max_iter:
                    raise ValueError(
                        f"The number of elements in the parameter {key} is inferior to max_iter."
                    )
        # If stepsize is a list of more than 1 element, backtracking is impossible.
        if len(params_algo["stepsize"]) > 1:
            if self.backtracking:
                self.backtracking = False
                raise Warning(
                    "Backtraking impossible when stepsize is predefined as a list. Setting backtrakcing to False."
                )

        # keep track of initial parameters in case they are changed during optimization (e.g. backtracking)
        self.init_params_algo = params_algo

        # By default, self.prior should be a list of elements of the class Prior. The user could want the prior to change at each iteration.
        if not isinstance(prior, Iterable):
            self.prior = [prior]
        else:
            self.prior = prior

        # Initialize the fixed-point module
        self.fixed_point = FixedPoint(
            iterator=iterator,
            update_params_fn=self.update_params_fn,
            update_prior_fn=self.update_prior_fn,
            check_iteration_fn=self.check_iteration_fn,
            check_conv_fn=self.check_conv_fn,
            init_metrics_fn=self.init_metrics_fn,
            init_iterate_fn=self.init_iterate_fn,
            update_metrics_fn=self.update_metrics_fn,
            max_iter=max_iter,
            early_stop=early_stop,
        )

    def update_params_fn(self, it):
        r"""
        For each parameter `params_algo`, selects the parameter value for iteration `it` (if this parameter depends on the iteration number).

        :param int it: iteration number.
        :return: a dictionary containing the parameters of iteration `it`.
        """
        cur_params_dict = {
            key: value[it] if len(value) > 1 else value[0]
            for key, value in zip(self.params_algo.keys(), self.params_algo.values())
        }
        return cur_params_dict

    def update_prior_fn(self, it):
        r"""
        For each prior function in `prior`, selects the prior value for iteration `it` (if this prior depends on the iteration number).

        :param int it: iteration number.
        :return: a dictionary containing the prior of iteration `it`.
        """
        prior_cur = self.prior[it] if len(self.prior) > 1 else self.prior[0]
        return prior_cur

    def get_primal_variable(self, X):
        r"""
        Returns the primal variable.

        :param dict X: dictionary containing the primal and auxiliary variables.
        :return: the primal variable.
        """
        return X["est"][0]

    def get_auxiliary_variable(self, X):
        r"""
        Returns the auxiliary variable.

        :param dict X: dictionary containing the primal and auxiliary variables.
        :return torch.Tensor X["est"][1]: the auxiliary variable.
        """
        return X["est"][1]

    def init_iterate_fn(self, y, physics, F_fn=None):
        r"""
        Initializes the iterate of the algorithm.
        The first iterate is stored in a dictionary of the form ``X = {'est': (x_0, u_0), 'cost': F_0}`` where:
            - `est` is a tuple containing the first primal and auxiliary iterates.
            - `cost` is the value of the cost function at the first iterate.

        By default, the first (primal, auxiliary) iterate of the algorithm is chosen as :math:`(A^*(y), A^*(y))`.
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
            x_init, z_init = physics.A_adjoint(y), physics.A_adjoint(y)
            init_X = self.custom_init(x_init, z_init)
        else:
            x_init, z_init = physics.A_adjoint(y), physics.A_adjoint(y)
            init_X = {"est": (x_init, z_init)}
        F = (
            F_fn(x_init, self.update_prior_fn(0), self.update_params_fn(0), y, physics)
            if self.has_cost and F_fn is not None
            else None
        )
        init_X["cost"] = F
        return init_X

    def init_metrics_fn(self, X_init, x_gt=None):
        r"""
        Initializes the metrics.
        Metrics are computed for each batch and for each iteration.
        They are represented by a list of list, and metrics[metric_name][i,j] contains the metric metric_name computed
        for batch i, at iteration j.

        :param dict X_init: dictionary containing the primal and auxiliary initial iterates.
        :param torch.Tensor x_gt: ground truth image, required for PSNR computation. Default: None.
        :return dict: A dictionary containing the metrics.
        """
        self.batch_size = self.get_primal_variable(X_init).shape[0]
        if self.return_metrics:
            init = {}
            x_init = (
                self.get_primal_variable(X_init)
                if not self.return_aux
                else self.get_auxiliary_variable(X_init)
            )
            if x_gt is not None:
                psnr = [[cal_psnr(x_init[i], x_gt[i])] for i in range(self.batch_size)]
            else:
                psnr = [[] for i in range(self.batch_size)]
            init["psnr"] = psnr
            if self.F_fn is not None:
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
            x_prev = (
                self.get_primal_variable(X_prev)
                if not self.return_aux
                else self.get_auxiliary_variable(X_prev)
            )
            x = (
                self.get_primal_variable(X)
                if not self.return_aux
                else self.get_auxiliary_variable(X)
            )
            for i in range(self.batch_size):
                residual = (
                    ((x_prev[i] - x[i]).norm() / (x[i].norm() + 1e-06))
                    .detach()
                    .cpu()
                    .item()
                )
                metrics["residual"][i].append(residual)
                if x_gt is not None:
                    psnr = cal_psnr(x[i], x_gt[i])
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
                                metrics[custom_metric_name], x_prev[i], x[i]
                            )
                        )
        return metrics

    def check_iteration_fn(self, X_prev, X):
        r"""
        Performs stepsize backtraking.

        :param dict X_prev: dictionary containing the primal and dual previous iterates.
        :param dict X: dictionary containing the current primal and dual iterates.
        """
        if self.backtracking and X_prev is not None:
            x_prev = self.get_primal_variable(X_prev)
            x = self.get_primal_variable(X)
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
        :param dict X_prev: dictionary containing the primal and dual previous iterates.
        :param dict X: dictionary containing the current primal and dual iterates.
        :return bool: `True` if the algorithm has converged, `False` otherwise.
        """
        if self.crit_conv == "residual":
            x_prev = (
                self.get_primal_variable(X_prev)
                if not self.return_aux
                else self.get_auxiliary_variable(X_prev)
            )
            x_prev = x_prev.reshape(x_prev.shape[0], -1)
            x = (
                self.get_primal_variable(X)
                if not self.return_aux
                else self.get_auxiliary_variable(X)
            )
            x = x.reshape(x.shape[0], -1)
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

    def forward(self, y, physics, x_gt=None):
        r"""
        Runs the fixed-point iteration algorithm for solving :ref:`(1) <optim>`.

        :param torch.Tensor y: measurement vector.
        :param deepinv.physics physics: physics of the problem for the acquisition of `y`.
        :param torch.Tensor x_gt: (optional) ground truth image, for plotting the PSNR across optim iterations.
        """
        x, metrics = self.fixed_point(y, physics, x_gt=x_gt)
        x = (
            self.get_primal_variable(x)
            if not self.return_aux
            else self.get_auxiliary_variable(x)
        )
        if self.return_metrics:
            return x, metrics
        else:
            return x


def optim_builder(
    iteration,
    data_fidelity=L2(),
    F_fn=None,
    g_first=False,
    beta=1.0,
    **kwargs,
):
    r"""
    Function building the appropriate Optimizer given its name.

    ::

        # Define the optimisation algorithm
        optim_algo = optim_builder(
                        'PGD',
                        prior=prior,
                        data_fidelity=data_fidelity,
                        max_iter=100,
                        crit_conv="residual",
                        thres_conv=1e-11,
                        verbose=True,
                        params_algo=params_algo,
                        early_stop=True,
                    )

        # Run the optimisation algorithm
        sol = optim_algo(y, physics)


    :param iteration: either name of the algorithm to be used, or an iterator.
        If an algorithm name (string), should be either `"PGD"`, `"ADMM"`, `"HQS"`, `"CP"` or `"DRS"`.
    :param dict params_algo: dictionary containing the algorithm's relevant parameter.
    :param deepinv.optim.data_fidelity data_fidelity: data fidelity term in the optimisation problem.
    :param F_fn: Custom user input cost function. default: None.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param float beta: relaxation parameter in the fixed point algorithm. Default: `1.0`.
    """
    # If no custom objective function F_fn is given but g is explicitly given, we have an explicit objective function.
    explicit_prior = (
        kwargs["prior"][0].explicit_prior
        if isinstance(kwargs["prior"], list)
        else kwargs["prior"].explicit_prior
    )
    if F_fn is None and explicit_prior:

        def F_fn(x, prior, cur_params, y, physics):
            return cur_params["lambda"] * data_fidelity(x, y, physics) + prior.g(
                x, cur_params["g_param"]
            )

        has_cost = True
    else:
        has_cost = False

    if isinstance(iteration, str):
        iterator_fn = str_to_class(iteration + "Iteration")
        iterator = iterator_fn(
            data_fidelity=data_fidelity,
            g_first=g_first,
            beta=beta,
            F_fn=F_fn,
            has_cost=has_cost,
        )
    else:
        iterator = iteration

    optimizer = BaseOptim(iterator, has_cost=has_cost, **kwargs)
    return optimizer


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
