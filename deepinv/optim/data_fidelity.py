import torch
import torch.nn as nn


class DataFidelity(nn.Module):
    r'''
    Data fidelity term for optimization algorithms.

    .. math:

        f(Ax,y)

    '''
    def __init__(self, f=None, grad_f=None, prox_f=None, prox_norm=None):
        super().__init__()
        self._grad_f = grad_f # TODO: use autograd?
        self._f = f
        self._prox_f = prox_f
        self._prox_norm = prox_norm

    def f(self, x, y):
        return self._f(x)

    def grad_f(self, x, y):
        return self._grad_f(x, y)

    def prox_f(self, x, y, gamma):
        return self._prox_f(x, y, gamma)

    def forward(self, x, y, physics):
        Ax = physics.A(x)
        return self.f(Ax, y)

    def grad(self, x, y, physics):
        Ax = physics.A(x)
        if self.grad_f is not None:
            return physics.A_adjoint(self.grad_f(Ax, y))
        else:
            raise ValueError('No gradient defined for this data fidelity term.')

    def prox(self, x, y, physics, stepsize):
        if ['Denoising'] in physics.__class__.__name__:
            return self.prox_f(y, x, stepsize)
        else:# TODO: use GD?
            raise Exception("no prox operator is implemented for the data fidelity term.")

    def prox_norm(self, x, y, stepsize):
        return self.prox_norm(x, y, stepsize)


class L2(DataFidelity):
    r'''
    L2 fidelity

    '''
    def __init__(self):
        super().__init__()

    def f(self, x, y):
        return (x-y).flatten().pow(2).sum()

    def grad_f(self, x, y):
        return x-y

    def prox(self, x, y, physics, stepsize):  # used to be in L2 but needs to be moved at the level of the data fidelity!!
        return physics.prox_l2(x, y, stepsize)

    def prox_f(self, x, y, gamma):  # Should be this instead?
        r'''
        computes the proximal operator of

        .. math::

            f(x) = \frac{1}{2}*\gamma*||x-y||_2^2

        '''
        return (x+gamma*y)/(1+gamma)


class IndicatorL2(DataFidelity):
    r'''
    Indicator of L2 ball with radius r

    '''
    def __init__(self, radius=None):
        super().__init__()
        self.radius = radius

    def f(self, x, y, radius=0.):
        dist = (x-y).flatten().pow(2).sum()
        loss = 0 if dist<radius else 1e16
        return loss

    def prox_f(self, x, y, gamma, radius=None):
        if radius is None:
            radius = self.radius
        return y + torch.min(torch.tensor([radius]), torch.linalg.norm(x.flatten() - y.flatten())) * (x - y) / \
            (torch.linalg.norm(x - y) + 1e-6)


class PoissonLikelihood(DataFidelity):
    r'''

    Poisson negative log likelihood

    '''
    def __init__(self, bkg=0):
        super().__init__()
        self.bkg = bkg

    def f(self, x, y):
        return (- y * torch.log(x + self.bkg)).flatten().sum() + x.flatten().sum()

    def grad_f(self, x, y):
        return - y/(x+self.bkg) + x.numel()

    def prox_f(self, x, y, gamma):
        out = x - 1/gamma * ((x-1/gamma).pow(2) + 4*y/gamma).sqrt()
        return out/2


class L1(DataFidelity):
    r'''
    L1 fidelity

    '''
    def __init__(self):
        super().__init__()

    def f(self, x, y):
        return (x - y).flatten().abs().sum()

    def grad_f(self, x, y):
        return torch.sign(x - y)

    def prox_f(self, x, y, gamma):
        # soft thresholding
        d = x-y
        aux = torch.sign(d)*torch.max(d.abs()-gamma, 0)
        return aux + y

