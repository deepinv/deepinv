from deepinv.optim_v2.base_optim import BaseOptim
import deepinv as dinv
from deepinv.optim.potential import Potential
from deepinv.models import Reconstructor

class PGD(BaseOptim): 

    def __init__(self, data_fidelity, prior, g_first = False, stepsize=1.0, lambda_reg=1.0, g_param=None, max_iter=100):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.g_first = g_first
        self.lambda_reg = lambda_reg
        self.g_param = g_param
        self.stepsize, self.max_iter = stepsize, max_iter

    def forward(self, y = None, physics = None, init=None, x_gt=None):
        if init is None:
            if y is None or physics is None:
                raise ValueError("If init is not provided, y and physics must be provided to compute the adjoint.")
            init = physics.A_adjoint(y)
        x = init
        for it in range(self.max_iter):
            if self.g_first:
                z = x - self.stepsize * self.data_fidelity.grad(x, y, physics)
                x = self.prior.prox(x, gamma=self.lambda_reg * self.stepsize)
            else:
                z = x - self.stepsize * self.lambda_reg * self.prior.grad(x, y, physics)
                x = self.prior.prox(z, gamma=self.stepsize)
        return x