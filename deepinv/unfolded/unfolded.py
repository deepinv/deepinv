import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.optim.utils import str_to_class
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
                if self.stepsize_iterable:
                    self.stepsize = nn.Parameter(self.stepsize[0].to(device))
                else :
                    self.stepsize = nn.Parameter(self.stepsize.to(device))
            else:
                if self.stepsize_iterable:
                    self.stepsize  = nn.ParameterList([nn.Parameter(self.stepsize[i].to(device)) for i in range(self.max_iter)])
                else :
                    self.stepsize = nn.ParameterList([nn.Parameter(self.stepsize.to(device)) for i in range(self.max_iter)])
                    self.stepsize_iterable = True

        if learn_g_param:
            if self.g_param is None : 
                self.g_param = 1.
            if constant_g_param:
                if self.g_param_iterable:
                    self.g_param = nn.Parameter(self.g_param[0].to(device))
                else :
                    self.g_param = nn.Parameter(self.g_param.to(device))
            else:
                if self.g_param_iterable:
                    self.g_param  = nn.ParameterList([nn.Parameter(self.g_param[i].to(device)) for i in range(self.max_iter)])
                else :
                    self.g_param = nn.ParameterList([nn.Parameter(self.g_param.to(device)) for i in range(self.max_iter)])
                    self.g_param_iterable = True
        
        if custom_g_step is not None:
            self.iterator.g_step = custom_g_step
        if custom_f_step is not None:
            self.iterator.f_step = custom_f_step



def Unfolded(algo_name, data_fidelity='L2', lamb=1., device='cpu', g=None, prox_g=None,
                 grad_g=None, g_first=False, stepsize=[1.] * 50, g_param=None, stepsize_inter=1.,
                 max_iter_inter=50, tol_inter=1e-3, beta=1., F_fn=None, **kwargs):
    iterator_fn = str_to_class(algo_name + 'Iteration')
    iterator = iterator_fn(data_fidelity=data_fidelity, lamb=lamb, device=device, g=g, prox_g=prox_g,
                 grad_g=grad_g, g_first=g_first, stepsize_inter=stepsize_inter,
                 max_iter_inter=max_iter_inter, tol_inter=tol_inter, beta=beta)
    return BaseUnfold(iterator, F_fn = F_fn,  g_param=g_param, stepsize=stepsize, **kwargs)