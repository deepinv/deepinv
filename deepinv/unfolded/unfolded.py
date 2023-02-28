import torch
import torch.nn as nn

from deepinv.optim.optim_iterator import PD
from deepinv.optim.fixed_point import FixedPoint

class Unfolded(nn.Module):
    '''
    Unfolded module
    '''
    def __init__(self, iterator, custom_prox_1=None, custom_prox_2=None, stepsize=1., max_iter=50, physics=None,
                 crit_conv=1e-3, verbose=True):
        super(Unfolded, self).__init__()

        self.max_iter = max_iter
        self.physics = physics

        self.iterator = iterator
        if custom_prox_1 is not None:
            self.iterator.primal_prox = nn.ModuleList([custom_prox_1])

        if custom_prox_2 is not None:
            self.iterator.dual_prox = nn.ModuleList([custom_prox_2])

        self.FP = FixedPoint(self.iterator, max_iter=max_iter, early_stop=True, crit_conv=crit_conv, verbose=verbose)
        # self.parameters = self.iterator.parameters

    def forward(self, x, physics=None):
        if physics is None:
            physics = self.physics
        x_init = (physics.A_adjoint(x), x)
        return self.FP(x_init, x, physics)

