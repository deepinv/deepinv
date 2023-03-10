import torch
import torch.nn as nn
from .optim_iterator import OptimIterator, fStep, gStep

class PGD(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: add doc
        '''
        super(PGD, self).__init__(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)


class fStepPGD(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepPGD, self).__init__(**kwargs)

    def forward(self, x, y, physics, it):
        if not self.g_first:
            return x - self.stepsize[it] * self.lamb * self.data_fidelity.grad(x, y, physics)
        else:
            return self.data_fidelity.prox(x, y, physics, self.lamb * self.stepsize[it])


class gStepPGD(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepPGD, self).__init__(**kwargs)

    def forward(self, x, it):
        if not self.g_first:
            return self.prox_g(x, self.g_param[it], it)
        else:
            return x - self.stepsize[it] * self.grad_g(x, self.g_param[it], it)


