import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.utils import str_to_class
from deepinv.optim.data_fidelity import L2
from collections.abc import Iterable


class BaseOptim(nn.Module):
    r"""
    Class for optimisation algorithms that iterates the iterator.

    iterator : ...

    :param deepinv.optim.iterator iterator: description

    """

    def __init__(
        self,
        iterator,
        params_algo={"lambda": 1.0, "stepsize": 1.0},
        prior=None,
        max_iter=50,
        crit_conv="residual",
        thres_conv=1e-5,
        early_stop=True,
        F_fn=None,
        anderson_acceleration=False,
        anderson_beta=1.0,
        anderson_history_size=5,
        verbose=False,
        return_dual=False,
        backtracking=False,
        gamma_backtracking=0.1,
        eta_backtracking=0.9,
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

        for key, value in zip(self.params_algo.keys(), self.params_algo.values()):
            if not isinstance(value, Iterable):
                self.params_algo[key] = [value]

        for key, value in zip(self.prior.keys(), self.prior.values()):
            if not isinstance(value, Iterable):
                self.prior[key] = [value]

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
                crit_conv=crit_conv,
                thres_conv=thres_conv,
                verbose=verbose,
            )
        else:
            self.fixed_point = FixedPoint(
                iterator,
                update_params_fn_pre=self.update_params_fn_pre,
                update_prior_fn=self.update_prior_fn,
                max_iter=max_iter,
                early_stop=early_stop,
                crit_conv=crit_conv,
                thres_conv=thres_conv,
                verbose=verbose,
            )

    def update_params_fn_pre(self, it, X, X_prev):
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
        cur_params_dict = {
            key: value[it] if len(value) > 1 else value[0]
            for key, value in zip(self.params_algo.keys(), self.params_algo.values())
        }
        return cur_params_dict

    def update_prior_fn(self, it):
        prior_cur = {
            key: value[it] if len(value) > 1 else value[0]
            for key, value in zip(self.prior.keys(), self.prior.values())
        }
        return prior_cur

    def get_init(self, cur_params, y, physics):
        r""" """
        x_init = physics.A_adjoint(y)
        init_X = {
            "est": (x_init, y),
            "cost": self.F_fn(x_init, cur_params, y, physics) if self.F_fn else None,
        }
        return init_X

    def get_primal_variable(self, X):
        return X["est"][0]

    def get_dual_variable(self, X):
        return X["est"][1]

    def forward(self, y, physics):
        init_params = self.get_params_it(0)
        x = self.get_init(init_params, y, physics)
        x = self.fixed_point(x, y, physics)
        return (
            self.get_primal_variable(x)
            if not self.return_dual
            else self.get_dual_variable(x)
        )

    def has_converged(self):
        return self.fixed_point.has_converged


def Optim(
    algo_name,
    params_algo,
    data_fidelity=L2(),
    F_fn=None,
    prior=None,
    g_first=False,
    beta=1.0,
    backtracking=False,
    gamma_backtracking=0.1,
    eta_backtracking=0.9,
    bregman_potential="L2",
    **kwargs
):
    iterator_fn = str_to_class(algo_name + "Iteration")
    iterator = iterator_fn(
        data_fidelity=data_fidelity,
        g_first=g_first,
        beta=beta,
        F_fn=F_fn,
        bregman_potential=bregman_potential,
    )
    optimizer = BaseOptim(
        iterator,
        params_algo=params_algo,
        prior=prior,
        F_fn=F_fn,
        backtracking=backtracking,
        gamma_backtracking=gamma_backtracking,
        eta_backtracking=eta_backtracking,
        **kwargs
    )
    return optimizer
