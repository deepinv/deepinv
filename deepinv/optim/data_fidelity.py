import torch
import torch.nn as nn


class DataFidelity(nn.Module):
    r'''
    Data fidelity term for optimization algorithms.

    .. math:

        f(Ax,y)

    where :math:

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

    def prox(self, x, y, physics, gamma):
        if 'Denoising' in physics.__class__.__name__:
            return self.prox_f(y, x, gamma)
        else:# TODO: use GD?
            raise Exception("no prox operator is implemented for the data fidelity term.")

    def prox_norm(self, x, y, gamma):
        return self.prox_norm(x, y, gamma)


class L2(DataFidelity):
    r'''
    L2 fidelity

    Describes the following data fidelity loss:

    .. math::

        f(x) = \frac{1}{2}\|x-y\|^2

    It can be used to define a log-likelihood function by setting a noise level
    '''
    def __init__(self, sigma=None):
        super().__init__()

        if sigma is not None:
            self.norm = 1/(sigma**2)
        else:
            self.norm = 1.

    def f(self, x, y):
        return self.norm*(x-y).flatten().pow(2).sum()/2

    def grad_f(self, x, y):
        return self.norm*(x-y)

    def prox(self, x, y, physics, gamma):  # used to be in L2 but needs to be moved at the level of the data fidelity!!
        return physics.prox_l2(x, y, self.norm*gamma)

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
    def __init__(self, gain=1., bkg=0, normalize=True):
        super().__init__()
        self.bkg = bkg
        self.gain = gain
        self.normalize = normalize

    def f(self, x, y):
        if self.normalize:
            y = y*self.gain
        return (- y * torch.log(self.gain*x + self.bkg)).flatten().sum() + (self.gain*x).flatten().sum()

    def grad_f(self, x, y):
        if self.normalize:
            y = y*self.gain
        return (1/self.gain)*(torch.ones_like(x) - y/(self.gain*x+self.bkg))

    def prox_f(self, x, y, gamma):
        if self.normalize:
            y = y*self.gain
        out = x - (self.gain/gamma) * ((x-self.gain/gamma).pow(2) + 4*y/gamma).sqrt()
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

