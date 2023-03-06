import torch
import torch.nn as nn

from deepinv.optim.optim_iterator import PD
from deepinv.optim.fixed_point import FixedPoint

class Unfolded(nn.Module):
    '''
    Unfolded module
    '''
    def __init__(self, iterator, stepsize=1., max_iter=50, physics=None, crit_conv=1e-3, learn_stepsize=False, learn_g_param=False, 
                 custom_g_step=None, custom_f_step=None, device=torch.device('cpu'), verbose=True, constant_stepsize=False, constant_g_param=False):
        super(Unfolded, self).__init__()

        self.max_iter = max_iter
        self.physics = physics
        self.device = device
        self.iterator = iterator
        self.crit_conv = crit_conv

        if learn_stepsize:
            if constant_stepsize : 
                self.step_size = nn.Parameter(torch.tensor(stepsize, device=self.device))
                self.stepsize_list = [self.step_size]*max_iter
            else :
                self.stepsize_list = nn.ParameterList([nn.Parameter(torch.tensor(stepsize, device=self.device))
                                               for i in range(max_iter)])
            self.iterator.stepsize = self.stepsize_list

        if learn_g_param:
            if constant_g_param :
                self.g_param = nn.Parameter(torch.tensor(g_param, device=self.device))
                self.g_param_list = [self.g_param]*max_iter
            else :
                self.g_param_list = nn.ParameterList([nn.Parameter(torch.tensor(g_param, device=self.device))
                                               for i in range(max_iter)])
            self.iterator.g_param = self.g_param_list

        self.custom_primal_prox = custom_primal_prox
        self.custom_dual_prox = custom_dual_prox

        if custom_g_step is not None:
            self.iterator.g_step = self.custom_g_step
        if custom_f_step is not None:
            self.iterator.f_step = self.custom_f_step

        self.FP = FixedPoint(self.iterator, max_iter=max_iter, early_stop=True, crit_conv=crit_conv, verbose=verbose)

    def forward(self, y, physics, **kwargs):

        x = self.iterator.get_init(y, physics)

        x_out = self.FP(x, y, physics, **kwargs)

        x_out = self.iterator.get_primal_variable(x_out)

        return x_out

    def primal_prox_step(self, x, Atu, it):
        return self.custom_primal_prox[it](x, Atu, it)

    def dual_prox_step(self, Ax_cur, u, y, it):
        return self.custom_dual_prox[it](Ax_cur, u, y, it)

    def stepsize(self, it):
        return self.stepsize_list[it]

    def sigma_denoiser(self, it):
        print(self.sigma_list[it])
        return self.sigma_list[it]