import torch
import torch.nn as nn
from .optim_iterator import OptimIterator

class PGD(OptimIterator):

    def __init__(self, **kwargs):
        '''
        In this case the algorithm works on the product space HxH^* so input/output variable is a concatenation of
        primal and dual variables.

        TODO:
        - check that there is no conflict with the data_fidelity.prox
        - check that there is freedom in how to apply replacement of prox operators (see J. Adler works)
        '''
        super(PGD, self).__init__(**kwargs)
        self.g_step = ProxGradGStep(**kwargs)
        self.f_step = ProxGradFStep(**kwargs)


class ProxGradFStep(nn.Module):

    def __init__(self, stepsize=None, lamb=1.0, data_fidelity=None, g_first=False,  **kwargs):
        """
        TODO: add doc
        """
        super(ProxGradFStep, self).__init__()
        self.stepsize = stepsize
        self.lamb = lamb
        self.data_fidelity = data_fidelity
        self.g_first = g_first

    def forward(self, x, y, physics, it):
        if not self.g_first:
            return x - self.stepsize[it] * self.lamb * self.data_fidelity.grad(x, y, physics)
        else:
            return self.data_fidelity.prox(x, y, physics, self.lamb * self.stepsize[it])


class ProxGradGStep(nn.Module):

    def __init__(self, prox_g=None, grad_g=None, g_param=None, stepsize=None, g_first=False, **kwargs):
        """
        TODO: add doc
        """
        super(ProxGradGStep, self).__init__()
        self.prox_g = prox_g
        self.grad_g = grad_g
        self.g_param = g_param
        self.stepsize = stepsize
        self.g_first = g_first

    def forward(self, x, it):
        if not self.g_first:
            return self.prox_g(x, self.g_param[it], it)
        else:
            return x - self.stepsize[it] * self.grad_g(x, self.g_param[it], it)


