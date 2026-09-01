from deepinv.optim.base_optim import BaseOptim
import deepinv as dinv
from deepinv.optim.potential import Potential
from deepinv.models import Reconstructor

class PGD(BaseOptim):
    def __init__(self, stepsize=1.0, max_iter=100):
        super().__init__()
        self.stepsize, self.max_iter = stepsize, max_iter

    def forward(self, x0, F: Potential, G: Potential):
        x = x0
        for it in range(self.max_iter):
            x = G.prox(x - self.stepsize * F.grad(x), gamma=self.stepsize)
        return x

class PGDReconstructor(Reconstructor):  

    def __init__(self, data_fidelity, prior, stepsize=1.0, lambda_reg=1.0, g_param=None, max_iter=100):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.lambda_reg = lambda_reg
        self.g_param = g_param
        self.stepsize, self.max_iter = stepsize, max_iter
        self.solver = PGD(stepsize, max_iter)

    def forward(self, y, physics, init=None):
        F = self.data_fidelity.bind(y, physics)
        G = (self.lambda_reg * self.prior).bind(self.g_param)
        x0 = init if init is not None else physics.A_adjoint(y)
        return self.solver(x0, F, G)