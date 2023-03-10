import torch
import torch.nn as nn
from .optim_iterator import OptimIterator

class PD(OptimIterator):

    def __init__(self, data_fidelity, stepsize_2=1., **kwargs):
        '''
        In this case the algorithm works on the product space HxH^* so input/output variable is a concatenation of
        primal and dual variables.

        TODO:
        - check that there is no conflict with the data_fidelity.prox
        - check that there is freedom in how to apply replacement of prox operators (see J. Adler works)
        '''
        super(PD, self).__init__(**kwargs)

        self.stepsize_2 = [stepsize_2/2.]*len(self.stepsize)
        self.data_fidelity = data_fidelity
        self.g_step = PrimalDualGStep(prox_g=self.prox_g, g_param=self.g_param, stepsize_2=self.stepsize_2)
        self.f_step = PrimalDualFStep(data_fidelity=self.data_fidelity, stepsize=self.stepsize, lamb=self.lamb)

    def forward(self, pd_var, it, y, physics):

        x, u = pd_var

        x_ = self.g_step(x, physics.A_adjoint(u), it)
        Ax_cur = physics.A(2 * x_ - x)
        u_ = self.f_step(Ax_cur, u, y, it)

        pd_variable = (x_, u_)

        return pd_variable


class PrimalDualFStep(nn.Module):

    def __init__(self, stepsize, lamb, data_fidelity):
        """
        TODO: add doc
        """
        super(PrimalDualFStep, self).__init__()

        self.stepsize = stepsize
        self.lamb = lamb
        self.data_fidelity = data_fidelity

    def forward(self, Ax_cur, u, y, it):  # Beware this is not the prox of f(A\cdot) but only the prox of f, A is tackled independently in PD
       v = u + self.stepsize[it] * Ax_cur
       return v - self.stepsize[it] * self.data_fidelity.prox_norm(v / self.stepsize[it], y, self.lamb)


class PrimalDualGStep(nn.Module):

    def __init__(self, prox_g=None, g_param=None, stepsize_2=None):
        """
        TODO: add doc
        """
        super(PrimalDualGStep, self).__init__()

        self.prox_g = prox_g
        self.g_param = g_param
        self.stepsize_2 = stepsize_2

    def forward(self, x, Atu, it):
        return self.prox_g(x - self.stepsize_2[it] * Atu, self.stepsize_2[it] * self.g_param[it], it)




