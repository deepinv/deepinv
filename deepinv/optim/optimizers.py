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

        for key, value in zip(self.prior.keys(), self.prior.values()):
            if not isinstance(value, Iterable):
                self.prior[key] = [value]

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
                update_params_fn_pre=self.update_params_fn_pre,
                update_prior_fn=self.update_prior_fn,
                max_iter=self.max_iter,
                history_size=anderson_history_size,
                beta=anderson_beta,
                early_stop=early_stop,
                check_conv_fn=self.check_conv_fn,
                init_metrics=self.init_metrics,
                update_metrics=self.update_metrics,
            )
        else:
            self.fixed_point = FixedPoint(
                iterator,
                update_params_fn_pre=self.update_params_fn_pre,
                update_prior_fn=self.update_prior_fn,
                max_iter=max_iter,
                early_stop=early_stop,
                check_conv_fn=self.check_conv_fn,
                init_metrics_fn=self.init_metrics_fn,
                update_metrics_fn=self.update_metrics_fn,
            )

    def update_params_fn_pre(self, it, X, X_prev):
        r"""
        Selects the parameters of the fixed-point algorithm before each iteration, with potential re-computation in the
        case of backtracking.

        :param int it: iteration number.
        :param dict X: current iterate.
        :param dict X_prev: previous iterate.
        :return: a dictionary containing the parameters of iteration `it`.
        """
        if self.backtracking and X_prev is not None:
            x_prev, x = X_prev["est"][0], X["est"][0]
            F_prev, F = X_prev["cost"], X["cost"]
            diff_F, diff_x = F_prev - F, (torch.norm(x - x_prev, p=2) ** 2).item()
            stepsize = self.params_algo["stepsize"][0]
            if diff_F < (self.gamma_backtracking / stepsize) * diff_x:
                self.params_algo["stepsize"] = [self.eta_backtracking * stepsize]
        cur_params = self.get_params_it(it)
        return cur_params

    def get_params_it(self, it):
        r"""
        Selects the appropriate algorithm parameters if the parameters depend on the iteration number.

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
        Selects the appropriate prior if the prior varies with the iteration number.

        :param int it: iteration number.
        :return: a dictionary containing the prior of iteration `it`.
        """
        prior_cur = {
            key: value[it] if len(value) > 1 else value[0]
            for key, value in zip(self.prior.keys(), self.prior.values())
        }
        return prior_cur

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
            self.F_fn(x_init, prior, cur_params, y, physics) if self.F_fn else None
        )
        init_X = {
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
        Initialises the metrics functions.

        :param dict X_init: dictionary containing the primal and dual initial iterates.
        :param torch.Tensor x_gt: ground truth image, required for PSNR computation. Default: None.
        :return dict: A dictionary containing the metrics.
        """
        if self.return_metrics:
            if not self.return_dual:
                psnr = [cal_psnr(self.get_primal_variable(X_init), x_gt)]
            else:
                psnr = []
            init = {"cost": [], "residual": [], "psnr": psnr}
            if self.custom_metrics is not None:
                for custom_metric_name in self.custom_metrics.keys():
                    init[custom_metric_name] = []
            return init

    def update_metrics_fn(self, metrics, X_prev, X, x_gt=None):
        r"""
        Updates the metrics functions.

        :param dict metrics: dictionary containing the metrics.
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
            residual = (x_prev - x).norm() / (x.norm() + 1e-06)
            metrics["residual"].append(residual.detach().cpu().item())
            if x_gt is not None:
                psnr = cal_psnr(x, x_gt)
                metrics["psnr"].append(psnr)
            if self.F_fn is not None:
                cost = X["cost"]
                metrics["cost"].append(cost.detach().cpu().item())
            if self.custom_metrics is not None:
                for custom_metric_name, custom_metric_fn in zip(
                    self.custom_metrics.keys(), self.custom_metrics.values()
                ):
                    metrics[custom_metric_name].append(
                        custom_metric_fn(metrics[custom_metric_name], X_prev, X)
                    )
        return metrics

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
            x = (
                self.get_primal_variable(X)
                if not self.return_dual
                else self.get_dual_variable(X)
            )
            crit_cur = (x_prev - x).norm() / (x.norm() + 1e-06)
        elif self.crit_conv == "cost":
            F_prev = X_prev["cost"]
            F = X["cost"]
            crit_cur = (F_prev - F).norm() / (F.norm() + 1e-06)
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
        init_params = self.get_params_it(0)
        init_pior = self.update_prior_fn(0)
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


def optimbuilder(
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
    :param F_fn: cost function. default: None.
    :param dict prior: dictionary containing the regularisation prior under the form of a denoiser, proximity operator,
                       gradient, or simply an auto-differentiable function.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param float beta: relaxation parameter in the fixed point algorithm. Default: `1.0`.
    :param bool backtracking: whether to apply a backtracking for stepsize selection. Default: `False`.
    :param float gamma_backtracking: :math:`\gamma` parameter in the backtracking selection. Default: `0.1`.
    :param float eta_backtracking: :math:`\eta` parameter in the backtracking selection. Default: `0.9`.
    :param str bregman_potential: default: `"L2"`
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
