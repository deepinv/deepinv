import torch
import torch.nn as nn

from deepinv.optim.optim_iterator import PD
from deepinv.optim.fixed_point import FixedPoint

class Unfolded(nn.Module):
    '''
    Unfolded module
    '''
    def __init__(self, iterator, custom_primal_prox=None, custom_dual_prox=None, stepsize=1., max_iter=50, physics=None,
                 crit_conv=1e-3, verbose=True):
        super(Unfolded, self).__init__()

        self.max_iter = max_iter
        self.physics = physics

        self.iterator = iterator

        self.custom_primal_prox = custom_primal_prox
        self.custom_dual_prox = custom_dual_prox

        if custom_primal_prox is not None:
            self.iterator._primal_prox = self.primal_prox_step
        if custom_dual_prox is not None:
            self.iterator._dual_prox = self.dual_prox_step

        self.FP = FixedPoint(self.iterator, max_iter=max_iter, early_stop=True, crit_conv=crit_conv, verbose=verbose)

    def forward(self, y, physics, **kwargs):
        x_init = (physics.A_adjoint(y), y)
        return self.FP(x_init, y, physics)

    def primal_prox_step(self, x, Atu, it):
        return self.custom_primal_prox[it](x, Atu, it)

    def dual_prox_step(self, Ax_cur, u, y, it):
        return self.custom_dual_prox[it](Ax_cur, u, y, it)