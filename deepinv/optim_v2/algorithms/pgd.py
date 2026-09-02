from deepinv.optim_v2.base_optim import BaseOptim
from deepinv.optim_v2.metrics import MetricsRecorder
import deepinv as dinv
from deepinv.optim.potential import Potential
from deepinv.models import Reconstructor


class PGD(BaseOptim):

    def __init__(
        self,
        data_fidelity,
        prior,
        g_first=False,
        stepsize=1.0,
        lambda_reg=1.0,
        g_param=None,
        max_iter=100,
        early_stop=False,
        tol=1e-5,
        crit_conv: str = "residual",
        custom_metrics=None,
    ):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.g_first = g_first
        self.lambda_reg = lambda_reg
        self.g_param = g_param
        self.stepsize, self.max_iter = stepsize, max_iter
        self.early_stop = early_stop
        self.tol = tol
        self.crit_conv = crit_conv
        self.custom_metrics = custom_metrics

    def step(self, x, y, physics):
        if self.g_first:
            z = x - self.stepsize * self.data_fidelity.grad(x, y, physics)
            x = self.prior.prox(x, gamma=self.lambda_reg * self.stepsize)
        else:
            z = x - self.stepsize * self.lambda_reg * self.prior.grad(x, y, physics)
            x = self.prior.prox(z, gamma=self.stepsize)
        return x

    def forward(
        self,
        y=None,
        physics=None,
        init=None,
        x_gt=None,
        compute_metrics=False,
        verbose=False,
    ):
        if init is None:
            if y is None or physics is None:
                raise ValueError(
                    "If init is not provided, y and physics must be provided to compute the adjoint."
                )
            init = physics.A_adjoint(y)
        x = init
        tau = self.stepsize
        F = self.objective(x, y, physics) if self.has_cost else None
        metrics = MetricsRecorder(x, x_gt, self.custom_metrics, compute_metrics)
        for it in range(self.max_iter):
            x_old, F_old = x, F
            x = self.step(x, y, physics, stepsize=tau)
            x, tau, F = (
                self.backtrack(x_old, x, tau, y, physics, F_prev=F_old, F=F)
                if self.backtracking is not None
                else (x, tau, F)
            )
            metrics.update(x_old, x)
            if self.early_stop and self._converged(it, x_old, x):
                if verbose:
                    print(f"Algorithm converged at iteration {it}.")
                break
        return (x, metrics.as_dict()) if compute_metrics else x
