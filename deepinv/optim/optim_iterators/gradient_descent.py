import torch
import torch.nn as nn
from .optim_iterator import OptimIterator

class GDIteration(OptimIterator):  # TODO

    def __init__(self, **kwargs):
        super(GD, self).__init__(**kwargs)

    def forward(self, x, it, y, physics):
        x = x[0]
        x = x - self.stepsize[it] * (
                    self.lamb * self.data_fidelity.grad(x, y, physics) + self.grad_g(x, g_param[it], it))
        return (x, )
