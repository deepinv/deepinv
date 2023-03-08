import torch
import torch.nn as nn

class FStep(nn.Module):

    def __init__(self, data_fidelity):
        """
        TODO: add doc
        """
        super(FStep, self).__init__()

        self.data_fidelity = data_fidelity

    def forward(self, y, physics, it):
        pass


class PDFStep(FStep):

    def __init__(self, stepsize=None, lamb=None, **kwargs):
        super(PDFStep, self).__init__(**kwargs)

        self.stepsize = stepsize
        self.lamb = lamb

    def forward(self, Ax_cur, u, y, it):  # Beware this is not the prox of f(A\cdot) but only the prox of f, A is tackled independently in PD
       v = u + self.stepsize[it] * Ax_cur
       return v - self.stepsize[it] * self.data_fidelity.prox_norm(v / self.stepsize[it], y, self.lamb)


class GStep(nn.Module):

    def __init__(self, prox_g, g_param, **kwargs):
        """
        TODO: add doc
        """
        super(GStep, self).__init__()

        self.prox_g = prox_g
        self.g_param = g_param

    def forward(self, x, it):
        pass


class PDGStep(GStep):

    def __init__(self, stepsize_2=None, **kwargs):
        super(PDGStep, self).__init__(**kwargs)

        self.stepsize_2 = stepsize_2

    def forward(self, x, Atu, it):
        return self.prox_g(x - self.stepsize_2[it] * Atu, self.stepsize_2[it] * self.g_param[it], it)


