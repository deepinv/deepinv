from deepinv.optim.base_optim import BaseOptim

class PGD(BaseOptim):
    def __init__(self, F, G, stepsize):
        super().__init__(F)
        self.stepsize = stepsize
        self.F = F
        self.G = G

    def forward(self, y, physics, init=None, x_gt=None):
        x = x - self.stepsize * self.F.grad(x, y, physics)
        x = self.G.prox(x, gamma=self.stepsize)
        return x