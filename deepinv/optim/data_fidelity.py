import torch
import torch.nn as nn

class data_fidelity(nn.Module):
    '''
    Data fidelity term for optimization algorithms.
    '''

    def __init__(self, type):
        super(optim, self).__init__()

        if type == 'L2':
            self.f = lambda x,y : 0.5 * (x-y).norm()**2
            self.grad = lambda x,y : x-y
        elif type == 'L1':
            self.f = lambda x,y : (x-y).abs().sum()
            self.grad = lambda x,y : torch.sign(x-y)
        elif type == 'KL':
            pass
        else :
            raise ValueError('Unknown data fidelity type.')

    def forward(self, x, y, physics):
        Ax = physics.A(x)
        return self.f(Ax,y)

    def grad(self, x, y, physics):
        Ax = physics.A(x)
        if self.grad is not None :
            return self.grad(Ax,y)
        else :
            raise ValueError('No gradient defined for this data fidelity term.')

    def prox(self, x, y, physics, stepsize):
        Ax = physics.A(x)
        return physics.L2_prox(Ax, y, stepsize)

