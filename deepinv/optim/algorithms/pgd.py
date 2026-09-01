from deepinv.optim.base_optim import BaseOptim
import deepinv as dinv
from deepinv.optim.potential import Potential

class PGD(BaseOptim):
    def __init__(self, F : Potential, G : Potential, stepsize : float, max_iter : int):
        super().__init__(F)
        self.stepsize = stepsize
        self.F = F
        self.G = G
        self.max_iter = max_iter

    def forward(self, init=None):
        x = init
        for it in range(self.max_iter):
            x = x - self.stepsize * self.F.grad(x)
            x = self.G.prox(x, gamma=self.stepsize)
        return x