import torch
import torch.nn as nn


class DataFidelity(nn.Module):
    r'''
    Data fidelity term for optimization algorithms.

    .. math:

        \datafid{Ax}{y}

    where ... TODO

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
        r'''
        Computes the data fidelity :math:`\datafid{Ax}{y}`.

        '''
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
    :math:`\ell_2` fidelity.

    Describes the following data fidelity loss:

    .. math::

        f(x) = \frac{1}{2\sigma^2}\|x-y\|^2

    It can be used to define a log-likelihood function associated with additive Gaussian noise
    by setting an appropriate noise level :math:`\sigma`.

    :param float sigma: Standard deviation of the noise.
    '''
    def __init__(self, sigma=1):
        super().__init__()

        self.sigma2 = 1/(sigma**2)

    def f(self, x, y):
        return self.sigma2*(x-y).flatten().pow(2).sum()/2

    def grad_f(self, x, y):
        return self.sigma2*(x-y)

    def prox(self, x, y, physics, stepsize):  # used to be in L2 but needs to be moved at the level of the data fidelity!!
        return physics.prox_l2(x, y, self.sigma2*stepsize)

    def prox_f(self, x, y, gamma):  # Should be this instead?
        r'''
        computes the proximal operator of

        .. math::

            f(x) = \frac{1}{2\sigma^2}\gamma\|x-y\|_2^2

        '''
        return (x+gamma*y)/(1+gamma) # TODO: fix sigma


class IndicatorL2(DataFidelity):
    r'''
    Indicator of :math:`\ell_2` ball with radius :math:`r`.

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

    Poisson negative log-likelihood.

    .. math::

        \datafid{z}{y} =  -y^{\top} \log(z+\beta)+1^{\top}z

    where :math:`y` are the measurements, :math:`z` is the estimated (positive) density and :math:`\beta\geq 0` is
    an optional background level.

    .. note::

        The function is not Lipschitz smooth w.r.t. :math:`z` in the absence of background (:math:`\beta=0`).

    :param float bkg: background level :math:`\beta`.
    '''
    def __init__(self, bkg=0.):
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
    :math:`\ell_1` fidelity.

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

