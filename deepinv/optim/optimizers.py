import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.optim import str_to_class

class BaseOptim(nn.Module):
    '''
    Class for optimisation algorithms that iterates the iterator.
        iterator : ...
    '''
    def __init__(self, iterator, max_iter=50, crit_conv=1e-3, early_stop=True, 
                anderson_acceleration=False, anderson_beta=1., anderson_history_size=5, verbose=False):
        super(BaseOptim, self).__init__()

        self.early_stop = early_stop
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.max_iter = max_iter
        self.anderson_acceleration = anderson_acceleration

        self.iterator = iterator

        if self.anderson_acceleration :
            self.anderson_beta = anderson_beta
            self.anderson_history_size = anderson_history_size
            self.fixed_point = AndersonAcceleration(self.iterator, max_iter=self.max_iter, history_size=anderson_history_size, beta=anderson_beta,
                            early_stop=early_stop, crit_conv=crit_conv, verbose=verbose)
        else :
            self.fixed_point = FixedPoint(self.iterator, max_iter=max_iter, early_stop=early_stop, crit_conv=crit_conv, verbose=verbose)

    def get_init(self, y, physics):
        return physics.A_adjoint(y), y

    def get_primal_variable(self, x):
        return x[0]

    def forward(self, y, physics, **kwargs):
        x = self.get_init(y, physics)
        x = self.fixed_point(x, y, physics, **kwargs)
        x = self.get_primal_variable(x)
        return x

    def has_converged(self):
        return self.fixed_point.has_converged

def Optim(algo_name, data_fidelity='L2', lamb=1., device='cpu', g=None, prox_g=None,
            grad_g=None, g_first=False, stepsize=[1.] * 50, g_param=None, stepsize_inter=1.,
            max_iter_inter=50, tol_inter=1e-3, beta=1., **kwargs):
    iterator_fn = str_to_class(algo_name + 'Iteration')
    iterator = iterator_fn(data_fidelity=data_fidelity, lamb=lamb, device=device, g=g, prox_g=prox_g,
                 grad_g=grad_g, g_first=g_first, stepsize=stepsize, g_param=g_param, stepsize_inter=stepsize_inter,
                 max_iter_inter=max_iter_inter, tol_inter=tol_inter, beta=beta)
    return BaseOptim(iterator, **kwargs)