import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.unfolded import str_to_class
from deepinv.optim.optimizers import BaseOptim

class BaseUnfold(BaseOptim):
    '''
    Unfolded module
    '''

    def __init__(self, *args, learn_stepsize=True, learn_g_param=False, custom_g_step=None, 
                 custom_f_step=None, constant_stepsize=False, constant_g_param=False, device=torch.device('cpu'), **kwargs):
        super(BaseUnfold, self).__init__(*args, **kwargs)

        if learn_stepsize:
            if constant_stepsize:
                step_size_f = nn.Parameter(torch.tensor(self.iterator.f_step.stepsize[0], device=device))
                stepsize_list_f = [step_size_f] * self.max_iter
            else:
                stepsize_list_f = nn.ParameterList([nn.Parameter(torch.tensor(self.iterator.f_step.stepsize[i], device=device))
                                                       for i in range(self.max_iter)])
            self.iterator.f_step.stepsize = stepsize_list_f

            if hasattr(self.iterator.g_step, 'stepsize'):  # For primal-dual where g_step also has a stepsize
                if constant_stepsize:
                    step_size_g = nn.Parameter(torch.tensor(self.iterator.g_step.stepsize[0], device=device))
                    stepsize_list_g = [step_size_g] * self.max_iter
                else:
                    stepsize_list_g = nn.ParameterList(
                        [nn.Parameter(torch.tensor(self.iterator.g_step.stepsize[i], device=device))
                         for i in range(self.max_iter)])
                self.iterator.g_step.stepsize = stepsize_list_g

        if learn_g_param:
            if constant_g_param:
                g_param = nn.Parameter(torch.tensor(self.iterator.g_step.g_param[0], device=device))
                g_param_list = [g_param] * self.max_iter
            else:
                g_param_list = nn.ParameterList([nn.Parameter(torch.tensor(self.iterator.g_step.g_param[i], device=device))
                                                      for i in range(self.max_iter)])
            self.iterator.g_step.g_param = g_param_list

        if custom_g_step is not None:
            self.iterator.g_step = custom_g_step
        if custom_f_step is not None:
            self.iterator.f_step = custom_f_step


def Unfolded(algo_name, data_fidelity='L2', lamb=1., device='cpu', g=None, prox_g=None,
                 grad_g=None, g_first=False, stepsize=[1.] * 50, g_param=None, stepsize_inter=1.,
                 max_iter_inter=50, tol_inter=1e-3, beta=1., **kwargs):
    iterator_fn = str_to_class(algo_name + 'Iteration')
    iterator = iterator_fn(data_fidelity=data_fidelity, lamb=lamb, device=device, g=g, prox_g=prox_g,
                 grad_g=grad_g, g_first=g_first, stepsize=stepsize, g_param=g_param, stepsize_inter=stepsize_inter,
                 max_iter_inter=max_iter_inter, tol_inter=tol_inter, beta=beta)
    return BaseUnfold(iterator, **kwargs)