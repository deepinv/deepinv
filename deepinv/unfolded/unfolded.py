import torch
import torch.nn as nn

from deepinv.optim.optim_iterator import PD
from deepinv.optim.fixed_point import FixedPoint

class Unfolded(nn.Module):
    '''
    Unfolded module
    '''
    def __init__(self, iterator, custom_primal_prox=None, custom_dual_prox=None, stepsize=1., max_iter=50, physics=None,
                 crit_conv=1e-3, verbose=True, device='cpu', learn_stepsizes=False, learn_sigmas=False):
        super(Unfolded, self).__init__()

        self.max_iter = max_iter
        self.physics = physics
        self.device = device

        self.iterator = iterator
        if learn_stepsizes:
            self.stepsize_list = nn.ParameterList([nn.Parameter(torch.tensor(1., device=self.device))
                                               for i in range(max_iter)])
            self.iterator.stepsize = self.stepsize

        if learn_sigmas:
            self.sigma_list = nn.ParameterList([nn.Parameter(torch.tensor(2., device=self.device))
                                               for i in range(max_iter)])
            self.iterator.sigma_denoiser = self.sigma_denoiser



        self.custom_primal_prox = custom_primal_prox
        self.custom_dual_prox = custom_dual_prox

        if custom_primal_prox is not None:
            self.iterator._primal_prox = self.primal_prox_step
        if custom_dual_prox is not None:
            self.iterator._dual_prox = self.dual_prox_step

        self.FP = FixedPoint(self.iterator, max_iter=max_iter, early_stop=True, crit_conv=crit_conv, verbose=verbose)

    def forward(self, y, physics, **kwargs):
        x_out = self.FP(y, physics)
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