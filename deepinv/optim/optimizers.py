import torch
import torch.nn as nn

from .fixed_point import FixedPoint
from .optim_iterators import ADMMIteration


class ADMM(nn.Module):
    '''
    ADMM algorithm.

        iterator : ...
    '''
    def __init__(self, data_fidelity='L2', lamb=1., device='cpu', g=None, prox_g=None,
                 grad_g=None, g_first=False, stepsize=[1.] * 50, g_param=None, stepsize_inter=1.,
                 max_iter_inter=50, tol_inter=1e-3, beta=1., max_iter=50,
                 crit_conv=1e-5, verbose=False, early_stop=False):
        super().__init__()

        self.iterator = ADMMIteration(data_fidelity=data_fidelity, lamb=lamb, device=device, g=g, prox_g=prox_g,
                 grad_g=grad_g, g_first=g_first, stepsize=stepsize, g_param=g_param, stepsize_inter=stepsize_inter,
                 max_iter_inter=max_iter_inter, tol_inter=tol_inter, beta=beta)

        self.fixed_point = FixedPoint(self.iterator, max_iter=max_iter, early_stop=early_stop, crit_conv=crit_conv,
                                      verbose=verbose)

        self.max_iter = max_iter
        self.crit_conv = crit_conv
        self.verbose = verbose
        # self.early_stop = early_stop

    def get_init(self, y, physics):
        return physics.A_adjoint(y), y

    def get_primal_variable(self, x):
        return x[0]

    def forward(self, y, physics, **kwargs):
        x = self.get_init(y, physics)
        x = self.fixed_point(x, y, physics, **kwargs)
        x = self.get_primal_variable(x)
        return x




