import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.utils import str_to_class
from deepinv.optim.data_fidelity import L2
from collections.abc import Iterable
from deepinv.utils import cal_psnr
from deepinv.optim.utils import gradient_descent


class BaseOptim(nn.Module):
    r"""
    Class for optimization algorithms iterating the fixed-point iterator.

    Module solving the problem

    .. math::
        \begin{equation}
        \underset{x}{\arg\min} \quad \datafid{\forw{x}}{y} + \reg{x}
        \end{equation}


    where the first term :math:`f:\yset\times\yset \mapsto \mathbb{R}_{+}` enforces data-fidelity
    (:math:`y \approx A(x)`), the second term :math:`g:\xset\mapsto \mathbb{R}_{+}` acts as a regularization, and
    :math:`A:\xset\mapsto \yset` is the forward operator (see :meth:`deepinv.physics.Physics`).

    Optimization algorithms for minimising the problem above can be written as fixed point algorithms,
    i.e. for :math:`k=1,2,...`

    .. math::
        \qquad (x_{k+1}, u_{k+1}) = \operatorname{FixedPoint}(x_k, u_k, f, g, A, y, ...)


    where :math:`x_k` is a primal variable converging to the solution of the minimisation problem, and
    :math:`u_k` is a dual variable.


    ::

            # Generate the data
            x = torch.ones(1, 1, 1, 3)
            A = torch.Tensor([[2, 0, 0], [0, -0.5, 0], [0, 0, 1]])
            A_forward = lambda v: A @ v
            A_adjoint = lambda v: A.transpose(0, 1) @ v

            # Define the physics model associated to this operator and the data
            physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
            y = physics.A(x)

            # Select the data fidelity term
            data_fidelity = L2()

            # Specify the prior and the algorithm parameters
            model_spec = {"name": "waveletprior", "args": {"wv": "db8", "level": 3, "device": device}}
            prior = {"prox_g": Denoiser(model_spec)}
            params_algo = {"stepsize": 0.1, "g_param": 1.0}

            # Choose the iterator associated to a specific algorithm
            iterator = PGDIteration(data_fidelity=data_fidelity)

            # Iterate the iterator
            max_iter = 50
            for it in range(max_iter):
                X = iterator(X, params)

            # Return the solution
            sol = X["est"]
            cost = X["cost"]




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
    :param function F_fn: cost function to be minimised by the optimization algorithm. Default: `None`.
    :param bool anderson_acceleration: whether to use anderson acceleration or not. Default: `False`.
    :param float anderson_beta: :math:`\beta` parameter in the anderson accleration. Default: `1.0`.
    :param int anderson_history_size: size of the history in anderson acceleration. Default: `5`.
    :param bool verbose: whether to print relevant information of the algorithm during its run,
                         such as convergence criterion at each iterate. Default: `False`.
    :param bool return_dual: whether to return the dual variable or not at the end of the algorithm. Default: `False`.
    :param bool backtracking: whether to apply a backtracking for stepsize selection. Default: `False`.
    :param float gamma_backtracking: :math:`\gamma` parameter in the backtracking selection. Default: `0.1`.
    :param float eta_backtracking: :math:`\eta` parameter in the backtracking selection. Default: `0.9`.
    :param function custom_init:  intializes the algorithm with `custom_init(y)`. If `None` (default value) algorithm is initilialized with :math:`A^Ty`. Default: `None`.
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
        F_fn=None,
        anderson_acceleration=False,
        anderson_beta=1.0,
        anderson_history_size=5,
        verbose=False,
        return_dual=False,
        backtracking=False,
        gamma_backtracking=0.1,
        eta_backtracking=0.9,
        return_metrics=False,
        custom_metrics=None,
        stepsize_prox_inter=1.0,
        max_iter_prox_inter=50,
        tol_prox_inter=1e-3,
        custom_init=None,
    ):
        super(BaseOptim, self).__init__()

        self.early_stop = early_stop
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.max_iter = max_iter
        self.anderson_acceleration = anderson_acceleration
        self.F_fn = F_fn
        self.return_dual = return_dual
        self.params_algo = params_algo
        self.prior = prior
        self.backtracking = backtracking
        self.gamma_backtracking = gamma_backtracking
        self.eta_backtracking = eta_backtracking
        self.return_metrics = return_metrics
        self.has_converged = False
        self.thres_conv = thres_conv
        self.custom_metrics = custom_metrics
        self.custom_init = custom_init

        for key, value in zip(self.params_algo.keys(), self.params_algo.values()):
            if not isinstance(value, Iterable):
                self.params_algo[key] = [value]
            else:
                if len(self.params_algo[key]) < self.max_iter:
                    raise ValueError(
                        f"The number of elements in the parameter {key} is inferior to max_iter."
                    )

        self.init_params_algo = (
            self.params_algo.copy()
        )  # keep track of initial parameters in case they are changed during optimization (e.g. backtracking)

        for key, value in zip(self.prior.keys(), self.prior.values()):
            if not isinstance(value, Iterable):
                self.prior[key] = [value]

        if len(self.params_algo["stepsize"]) > 1:
            if self.backtracking:
                self.backtracking = False
                raise Warning(
                    "Backtraking impossible when stepsize is predefined as a list. Setting backtrakcing to False."
                )

        # handle priors without explicit prox or grad
        if (
            iterator.requires_prox_g and "prox_g" not in self.prior.keys()
        ) or iterator.requires_grad_g:
            # we need at least the grad
            if "grad_g" not in self.prior.keys():
                if "g" in self.prior.keys():
                    self.prior["grad_g"] = []
                    for g in self.prior["g"]:
                        assert isinstance(
                            g, nn.Module
                        ), "The given prior must be an instance of nn.Module"

                        def grad_g(x, *args):
                            torch.set_grad_enabled(True)
                            x = x.requires_grad_()
                            return torch.autograd.grad(
                                g(x, *args), x, create_graph=True, only_inputs=True
                            )[0]

                        self.prior["grad_g"].append(grad_g)
            if iterator.requires_prox_g and "prox_g" not in self.prior.keys():
                self.prior["prox_g"] = []
                for grad_g in self.prior["grad_g"]:

                    def prox_g(x, *args, gamma=1):
                        grad = lambda y: gamma * grad_g(y, *args) + (1 / 2) * (y - x)
                        return gradient_descent(
                            grad,
                            x,
                            stepsize_prox_inter,
                            max_iter=max_iter_prox_inter,
                            tol=tol_prox_inter,
                        )

                    self.prior["prox_g"].append(prox_g)

        if self.anderson_acceleration:
            self.anderson_beta = anderson_beta
            self.anderson_history_size = anderson_history_size
            self.fixed_point = AndersonAcceleration(
                iterator,
                update_params_fn=self.update_params_fn,
                update_prior_fn=self.update_prior_fn,
                max_iter=self.max_iter,
                history_size=anderson_history_size,
                beta=anderson_beta,
                early_stop=early_stop,
                check_iteration_fn=self.check_iteration_fn,
                check_conv_fn=self.check_conv_fn,
                init_metrics=self.init_metrics,
                update_metrics=self.update_metrics,
            )
        else:
            self.fixed_point = FixedPoint(
                iterator,
                update_params_fn=self.update_params_fn,
                update_prior_fn=self.update_prior_fn,
                max_iter=max_iter,
                early_stop=early_stop,
                check_iteration_fn=self.check_iteration_fn,
                check_conv_fn=self.check_conv_fn,
                init_metrics_fn=self.init_metrics_fn,
                update_metrics_fn=self.update_metrics_fn,
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

    def init_params_fn(self):
        r"""
        Initialize (or updates) the dictionary of parameters.
        This is necessary if the parameters have been updated during optimization, for example via backtracking.

        :return: a dictionary containing the parameters of iteration `0`.
        """

        # self.params_algo = self.init_params_algo.copy()
        init_params = {
            key: value[0]
            for key, value in zip(
                self.init_params_algo.keys(), self.init_params_algo.values()
            )
        }
        return init_params

    def update_prior_fn(self, it):
        r"""
        For each prior function in `prior`, selects the prior value for iteration `it` (if this prior depends on the iteration number).

        :param int it: iteration number.
        :return: a dictionary containing the prior of iteration `it`.
        """
        prior_cur = {
            key: value[it] if len(value) > 1 else value[0]
            for key, value in zip(self.prior.keys(), self.prior.values())
        }
        return prior_cur

    def init_prior_fn(self):
        r"""
        Get initialization prior.

        :return: a dictionary containing the parameters of iteration `0`.
        """
        return self.update_prior_fn(0)

    def get_init(self, prior, cur_params, y, physics):
        r"""
        Initialises the parameters of the algorithm.

        By default, the first (primal, dual) iterate of the algorithm is chosen as :math:`(A^*(y), y)`.

        :param dict cur_params: dictionary containing the parameters related to the optimisation problem.
        :param torch.Tensor y: measurement vector.
        :param deepinv.physics: physics of the problem.
        :return: a dictionary containing: `"est"`, the primal-dual initialised variables; `"cost"`: the initial cost function.
        """
        if self.custom_init:
            x_init = self.custom_init(y)
        else:
            x_init = physics.A_adjoint(y)
        cost_init = (
            torch.tensor(
                [
                    self.F_fn(
                        x_init[i].unsqueeze(0),
                        prior,
                        cur_params,
                        y[i].unsqueeze(0),
                        physics,
                    )
                    for i in range(len(x_init))
                ]
            )
            if self.F_fn
            else None
        )
        init_X = {  # TODO: naming is a bit weird
            "est": (x_init, x_init),
            "cost": cost_init,
        }
        return init_X

    def get_primal_variable(self, X):
        r"""
        Returns the primal variable.

        :param dict X: dictionary containing the primal and dual iterates.
        :return: the primal variable.
        """
        return X["est"][0]

    def get_dual_variable(self, X):
        r"""
        Returns the dual variable.

        :param dict X: dictionary containing the primal and dual iterates.
        :return torch.Tensor X["est"][1]: the dual variable.
        """
        return X["est"][1]

    def init_metrics_fn(self, X_init, x_gt=None):
        r"""
        Initialises the metrics.
        Metrics are computed for each batch and for each iteration.
        They are reprenseted by a list of list, and metrics[metric_name][i,j] constains the metric metric_name computed for batch i, at iteration j.

        :param dict X_init: dictionary containing the primal and dual initial iterates.
        :param torch.Tensor x_gt: ground truth image, required for PSNR computation. Default: None.
        :return dict: A dictionary containing the metrics.
        """
        self.batch_size = self.get_primal_variable(X_init).shape[0]
        if self.return_metrics:
            init = {}
            if not self.return_dual:
                x_init = self.get_primal_variable(X_init)
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
                if not self.return_dual
                else self.get_dual_variable(X_prev)
            )
            x = (
                self.get_primal_variable(X)
                if not self.return_dual
                else self.get_dual_variable(X)
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
                if self.F_fn is not None:
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
        Check that the previous iteration decreases the objective function and perform stepsize backtraking.

        :param dict X_prev: dictionary containing the primal and dual previous iterates.
        :param dict X: dictionary containing the current primal and dual iterates.
        """
        if self.backtracking and X_prev is not None:
            x_prev = self.get_primal_variable(X_prev)
            x = self.get_primal_variable(X)
            x_prev = x_prev.reshape((x_prev.shape[0], -1))
            x = x.reshape((x.shape[0], -1))
            F_prev, F = X_prev["cost"], X["cost"]
            diff_F, diff_x = (F_prev - F).mean(), (
                torch.norm(x - x_prev, p=2, dim=-1) ** 2
            ).mean()
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
                if not self.return_dual
                else self.get_dual_variable(X_prev)
            )
            x_prev = x_prev.view(x_prev.shape[0], -1)
            x = (
                self.get_primal_variable(X)
                if not self.return_dual
                else self.get_dual_variable(X)
            )
            x = x.view(x.shape[0], -1)
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
        """
        init_params = self.init_params_fn()
        init_pior = self.init_prior_fn()
        x = self.get_init(init_pior, init_params, y, physics)
        x, metrics = self.fixed_point(x, y, physics, x_gt=x_gt)
        x = (
            self.get_primal_variable(x)
            if not self.return_dual
            else self.get_dual_variable(x)
        )
        if self.return_metrics:
            return x, metrics
        else:
            return x


def optim_builder(
    algo_name,
    data_fidelity=L2(),
    F_fn=None,
    g_first=False,
    beta=1.0,
    bregman_potential="L2",
    prior={},
    **kwargs,
):
    r"""
    Function building the appropriate Optimizer.

    :param str algo_name: name of the algorithm to be used. Should be either `"PGD"`, `"ADMM"`, `"HQS"`, `"PD"` or `"DRS"`.
    :param dict params_algo: dictionary containing the algorithm's relevant parameter.
    :param deepinv.optim.data_fidelity data_fidelity: data fidelity term in the optimisation problem.
    :param F_fn: Custom user input cost function. default: None.
    :param dict prior: dictionary containing the regularisation prior under the form of a denoiser, proximity operator,
                       gradient, or simply an auto-differentiable function.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param float beta: relaxation parameter in the fixed point algorithm. Default: `1.0`.
    :param bool backtracking: whether to apply a backtracking for stepsize selection. Default: `False`.
    :param float gamma_backtracking: :math:`\gamma` parameter in the backtracking selection. Default: `0.1`.
    :param float eta_backtracking: :math:`\eta` parameter in the backtracking selection. Default: `0.9`.
    :param str bregman_potential: possibility to perform optimization with another bregman geometry. Default: `"L2"`
    """

    # If no custom objective function F_fn is given but g is explicitly given, we have an explicit objective function.
    if F_fn is None and "g" in prior.keys():
        F_fn = lambda x, prior, cur_params, y, physics: cur_params[
            "lambda"
        ] * data_fidelity.f(physics.A(x), y) + prior["g"](x, cur_params["g_param"])
    iterator_fn = str_to_class(algo_name + "Iteration")
    iterator = iterator_fn(
        data_fidelity=data_fidelity,
        g_first=g_first,
        beta=beta,
        F_fn=F_fn,
        bregman_potential=bregman_potential,
    )
    optimizer = BaseOptim(iterator, F_fn=F_fn, prior=prior, **kwargs)
    return optimizer
