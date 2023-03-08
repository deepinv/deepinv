import torch
import torch.nn as nn
from .optim_iterator import OptimIterator

class HQS(OptimIterator):

    def __init__(self, **kwargs):
        '''
        In this case the algorithm works on the product space HxH^* so input/output variable is a concatenation of
        primal and dual variables.

        TODO:
        - check that there is no conflict with the data_fidelity.prox
        - check that there is freedom in how to apply replacement of prox operators (see J. Adler works)
        '''
        super(HQS, self).__init__(**kwargs)
        self.g_step = HalfQuadGStep(**kwargs)
        self.f_step = HalfQuadFStep(**kwargs)


class HalfQuadFStep(nn.Module):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(HalfQuadFStep, self).__init__(**kwargs)

    def forward(self, x, y, physics, it):
        return self.data_fidelity.prox(x, y, physics, self.lamb * self.stepsize[it])


class HalfQuadGStep(nn.Module):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(HalfQuadGStep, self).__init__(**kwargs)

    def forward(self, z, it):
        return self.prox_g(z, self.g_param[it], it)


