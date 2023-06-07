import torch
import torch.nn as nn

from deepinv.optim.utils import gradient_descent


class Prior(nn.Module):
    r"""
    Prior term :math:`g{x}`.

    This is the base class for the prior term :math:`g{x}`.
    """

    def __init__(self, g=None):
        super().__init__()
        self._g = g

    def g(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self._g(x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.g(x, *args, **kwargs)

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the prior term :math:`g` at :math:`x`. 
        By default, the gradient is computed using automatic differentiation.

        :param torch.tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x g`, computed in :math:`x`.
        """
        torch.set_grad_enabled(True)
        x = x.requires_grad_()
        return torch.autograd.grad(
            self.g(x, *args, **kwargs), x, create_graph=True, only_inputs=True
        )[0]

    def prox(self, x, gamma, *args, stepsize_inter=1.0, max_iter_inter=50, tol_inter=1e-3, **kwargs):
        r"""
        Calculates the proximity operator of :math:`g` at :math:`x`. By default, the proximity operator is computed using internal gradient descent.

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma g}(x)`, computed in :math:`x`.
        """
        grad = lambda z: gamma * self.grad(z, *args, **kwargs) + (z - x)
        return gradient_descent(
            grad,
            x,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
        )


class Tikhonov(Prior):
    r"""
    Tikhonov regularizer :math:`g{x} = \frac{1}{2}\| T x \|_2^2`.
    """

    def __init__(self, T):
        self.T = T
        super().__init__()

    def g(self, x):
        r"""
        Computes the Tikhonov regularizer :math:`g(x) = \frac{1}{2}\|T(x)\|_2^2`.

        :param torch.tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return 0.5 * torch.norm(self.T * x) ** 2

    def grad(self, x):
        r"""
        Calculates the gradient of the Tikhonov regularization term :math:`g` at :math:`x`. 

        :param torch.tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient at :math:`x`.
        """
        return x

    def prox(self, x, gamma):
        r"""
        Calculates the proximity operator of the Tikhonov regularization term :math:`g` at :math:`x`. 

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.tensor) proximity operator at :math:`x`.
        """
        return (1/(gamma+1))*x

class PnP(Prior):
    r"""
    Plug-and-play prior :math:`\operatorname{prox}_{\gamma g}(x) = \operatorname{D}_{\sigma}(x)`
    """

    def __init__(self, denoiser):
        self.denoiser = denoiser
        super().__init__()

    def prox(self, x, gamma, *args, **kwargs):
        r"""
        Uses denoising as the proximity operator of the PnP prior :math:`g` at :math:`x`. 

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.tensor) proximity operator at :math:`x`.
        """
        return self.denoiser(x, *args, **kwargs)



class RED(Prior):
    r"""
    Regularization-by-Denoising (RED) prior :math:`\nabla g(x) = \operatorname{Id} - \operatorname{D}_{\sigma}(x)`
    """

    def __init__(self, denoiser):
        self.denoiser = denoiser
        super().__init__()

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the prior term :math:`g` at :math:`x`. 
        By default, the gradient is computed using automatic differentiation.

        :param torch.tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x g`, computed in :math:`x`.
        """
        return x - self.denoiser(x, *args, **kwargs)



        