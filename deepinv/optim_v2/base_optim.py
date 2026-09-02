import torch.nn as nn
from dataclasses import dataclass
import deepinv as dinv
import warnings
from deepinv.loss.metric import Metric


@dataclass
class BacktrackingConfig:
    """Configuration parameters for backtracking line search on the stepsize.

    :param  float gamma: Armijo-like parameter (controls sufficient decrease).
    :param  float eta: Step reduction factor (e.g. multiply step by eta on failure).
    :param  int max_iter: Maximum number of backtracking steps.
    """

    gamma: float = 0.1
    eta: float = 0.9
    max_iter: int = 20


class BaseOptim(nn.Module):

    def __init__(
        self,
        data_fidelity: dinv.optim.DataFidelity,
        prior: dinv.optim.Prior,
        stepsize: float = 1.0,
        lambda_reg: float = 1.0,
        g_param: float = None,
        max_iter: int = 100,
        early_stop: bool = False,
        tol: float = 1e-5,
        crit_conv: str = "residual",
        custom_metrics: dict[str, Metric] = None,
        backtracking: BacktrackingConfig = None,
    ):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.lambda_reg = lambda_reg
        self.g_param = g_param
        self.max_iter = max_iter
        self.has_cost = self.prior.explicit_prior
        self.crit_conv = crit_conv
        self.early_stop = early_stop
        self.tol = tol
        self.custom_metrics = custom_metrics
        if isinstance(backtracking, bool):
            self.backtracking = backtracking
            self.backtracking_config = BacktrackingConfig() if backtracking else None
        else:
            self.backtracking = backtracking is not None
            self.backtracking_config = backtracking or BacktrackingConfig()

        if not self.has_cost and self.backtracking:
            self.backtracking = None
            warnings.warn("Backtracking impossible when no cost function is given.")

    def step(self, x, y, physics):
        raise NotImplementedError

    def forward(self, y=None, physics=None, init=None, x_gt=None):
        raise NotImplementedError

    def objective(self, x, y, physics):
        if not self.has_cost:
            raise ValueError("Objective function is not defined for this optimizer.")
        return self.data_fidelity(x, y, physics) + self.lambda_reg * self.prior(x)

    def _converged(self, it, x_old=None, x_new=None, F_prev=None, F=None):
        if self.crit_conv == "residual":
            res = (x_new - x_old).norm() / x_old.norm()
        elif self.crit_conv == "objective" and F_prev is not None and F is not None:
            res = abs(F - F_prev) / abs(F_prev)
        else:
            raise ValueError(f"Unknown convergence criterion: {self.crit_conv}")
        return res < self.tol

    def backtrack(self, x_prev, x, stepsize, y, physics, F_prev=None, F=None):
        if not self.has_cost:
            return x, stepsize, None
        cfg = self.backtracking
        if cfg is None:
            return x, stepsize, None
        F_prev = self.objective(x_prev, y, physics) if F_prev is None else F_prev
        F = self.objective(x, y, physics) if F is None else F
        for _ in range(cfg.max_iter):
            diff_F = (F_prev - F).mean()
            diff_x = (x - x_prev).flatten(1).pow(2).sum(-1).mean()
            if diff_F >= (cfg.gamma / stepsize) * diff_x:
                break  # sufficient decrease: accept
            stepsize = cfg.eta * stepsize  # shrink
            x = self.step(
                x_prev, y, physics, stepsize=stepsize
            )  # recompute from x_prev
            F = self.objective(x, y, physics)
        return x, stepsize, F
