import torch
import torch.nn as nn
import sys
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.unfolded import str_to_class

class Unfolded(nn.Module):
    '''
    Unfolded module
    '''

    def __init__(self, iterator, max_iter=50, crit_conv=1e-3, learn_stepsize=True, learn_g_param=False,
                 custom_g_step=None, custom_f_step=None, constant_stepsize=False, constant_g_param=False, early_stop=True, 
                 anderson_acceleration=False, anderson_beta=1., anderson_history_size=5, device=torch.device('cpu'), verbose=False):
        super(Unfolded, self).__init__()

        self.early_stop = early_stop
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.max_iter = max_iter
        self.anderson_acceleration = anderson_acceleration
        
        # model parameters
        self.iterator = iterator

        if learn_stepsize:
            if constant_stepsize:
                step_size_f = nn.Parameter(torch.tensor(iterator.f_step.stepsize[0], device=device))
                stepsize_list_f = [step_size_f] * max_iter
            else:
                stepsize_list_f = nn.ParameterList([nn.Parameter(torch.tensor(iterator.f_step.stepsize[i], device=device))
                                                       for i in range(max_iter)])
            self.iterator.f_step.stepsize = stepsize_list_f

            if hasattr(self.iterator.g_step, 'stepsize'):  # For primal-dual where g_step also has a stepsize
                if constant_stepsize:
                    step_size_g = nn.Parameter(torch.tensor(iterator.g_step.stepsize[0], device=device))
                    stepsize_list_g = [step_size_g] * max_iter
                else:
                    stepsize_list_g = nn.ParameterList(
                        [nn.Parameter(torch.tensor(iterator.g_step.stepsize[i], device=device))
                         for i in range(max_iter)])
                self.iterator.g_step.stepsize = stepsize_list_g

        if learn_g_param:
            if constant_g_param:
                g_param = nn.Parameter(torch.tensor(iterator.g_step.g_param[0], device=device))
                g_param_list = [g_param] * max_iter
            else:
                g_param_list = nn.ParameterList([nn.Parameter(torch.tensor(iterator.g_step.g_param[i], device=device))
                                                      for i in range(max_iter)])
            self.iterator.g_step.g_param = g_param_list

        if custom_g_step is not None:
            self.iterator.g_step = custom_g_step
        if custom_f_step is not None:
            self.iterator.f_step = custom_f_step

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


def Unfolded_algo(algo_name, data_fidelity='L2', lamb=1., device='cpu', g=None, prox_g=None,
                 grad_g=None, g_first=False, stepsize=[1.] * 50, g_param=None, stepsize_inter=1.,
                 max_iter_inter=50, tol_inter=1e-3, beta=1., **kwargs):
    iterator_fn = str_to_class(algo_name + 'Iteration')
    iterator = iterator_fn(data_fidelity=data_fidelity, lamb=lamb, device=device, g=g, prox_g=prox_g,
                 grad_g=grad_g, g_first=g_first, stepsize=stepsize, g_param=g_param, stepsize_inter=stepsize_inter,
                 max_iter_inter=max_iter_inter, tol_inter=tol_inter, beta=beta)
    return Unfolded(iterator, **kwargs)