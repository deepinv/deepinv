from deepinv.optim.base_optim import BaseOptim

class GD(BaseOptim):
    def __init__(self, F, stepsize):
        super().__init__(F)
        self.stepsize = stepsize
        self.F = F

    def forward(self, y, physics, init=None, x_gt=None):
        x = x - self.stepsize * self.F.grad(x, y, physics)
        return x
        
