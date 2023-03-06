import torch
import torch.nn as nn

from deepinv.optim.optim_iterator import PD
from deepinv.optim.fixed_point import FixedPoint

class Unfolded(nn.Module):
    '''
    Unfolded module
    '''
    def __init__(self, iterator, init=None, max_iter=50, crit_conv=1e-3, learn_stepsize=False, learn_g_param=False, 
                 custom_g_step=None, custom_f_step=None, device=torch.device('cpu'), verbose=True, constant_stepsize=False, constant_g_param=False):
        super(Unfolded, self).__init__()

        self.max_iter = max_iter
        self.device = device
        self.iterator = iterator
        self.crit_conv = crit_conv

        if learn_stepsize:
            if constant_stepsize : 
                self.step_size = nn.Parameter(torch.tensor(iterator.stepsize[0], device=self.device))
                self.stepsize_list = [self.step_size]*max_iter
            else :
                self.stepsize_list = nn.ParameterList([nn.Parameter(torch.tensor(iterator.stepsize[i], device=self.device))
                                               for i in range(max_iter)])
            self.iterator.stepsize = self.stepsize_list

        if learn_g_param:
            if constant_g_param :
                self.g_param = nn.Parameter(torch.tensor(iterator.g_param[0], device=self.device))
                self.g_param_list = [self.g_param]*max_iter
            else :
                self.g_param_list = nn.ParameterList([nn.Parameter(torch.tensor(iterator.g_param[i], device=self.device))
                                               for i in range(max_iter)])
            self.iterator.g_param = self.g_param_list

        if custom_g_step is not None:
            self.iterator.g_step = self.custom_g_step # COMMENT : can we avoid the 'primal_prox_step' fct by asking custom_g_step to take the same args as g_step and f_step ?
        if custom_f_step is not None:
            self.iterator.f_step = self.custom_f_step

        self.FP = FixedPoint(self.iterator, max_iter=max_iter, early_stop=True, crit_conv=crit_conv, verbose=verbose)

    def get_init(self, y, physics):
        return physics.A_adjoint(y), y

    def get_primal_variable(self, x):
        return x[0]

    def forward(self, y, physics, **kwargs):
        x = self.get_init(y, physics)
        x = self.FP(x, y, physics, **kwargs)
        x = self.get_primal_variable(x)
        return x