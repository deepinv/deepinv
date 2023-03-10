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
        self.g_step = DRSFStep(**kwargs)
        self.f_step = DRSGStep(**kwargs)


class DRSFStep(nn.Module):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(DRSFStep, self).__init__(**kwargs)

    def forward(self, x, y, physics, it):
        return 2 * self.data_fidelity.prox(x, y, physics, self.lamb * self.stepsize[it]) - x


class DRSGStep(nn.Module):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(DRSGStep, self).__init__(**kwargs)

    def forward(self, z, it):
        return 2 * self.prox_g(z, self.g_param[it], it) - z


