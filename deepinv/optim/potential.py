import torch
import torch.nn as nn
from deepinv.optim.utils import gradient_descent


class Potential(nn.Module):
    r"""
    Base class for a potential :math:`h : \xset \to \mathbb{R}` to be used in an optimization problem.

    Comes with methods to compute the potential gradient, its proximity operator, its convex conjugate (and associated gradient and prox).

    :param Callable fn: Potential function :math:`h(x)` to be used in the optimization problem.
    """

    def __init__(self, fn=None):
        super().__init__()
        self._fn = fn

    def fn(self, x, *args, **kwargs):
        r"""
        Computes the value of the potential :math:`h(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) prior :math:`h(x)`.
        """
        return self._fn(x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the value of the potential :math:`h(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) prior :math:`h(x)`.
        """
        return self.fn(x, *args, **kwargs)

    def conjugate(self, x, *args, **kwargs):
        r"""
        Computes the convex conjugate potential :math:`h^*(y) = \sup_{x} \langle x, y \rangle - h(x)`.
        By default, the conjugate is computed using internal gradient descent.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`h^*(y)`.
        """
        grad = lambda z: self.grad(z, *args, **kwargs) - x
        z = gradient_descent(-grad, x)
        return self.forward(z, *args, **kwargs) - torch.sum(
            x.reshape(x.shape[0], -1) * z.reshape(z.shape[0], -1), dim=-1
        ).view(x.shape[0], 1)

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the potential term :math:`h` at :math:`x`.
        By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h`, computed in :math:`x`.
        """
        with torch.enable_grad():
            x = x.requires_grad_()
            h = self.forward(x, *args, **kwargs)
            grad = torch.autograd.grad(
                h, x, torch.ones_like(h), create_graph=True, only_inputs=True
            )[0]
        return grad

    def grad_conj(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the convex conjugate potential :math:`h^*` at :math:`x`.
        If the potential is convex and differentiable, the gradient of the conjugate is the inverse of the gradient of the potential.
        By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h^*`, computed in :math:`x`.
        """
        with torch.enable_grad():
            x = x.requires_grad_()
            h = self.conjugate(x, *args, **kwargs)
            grad = torch.autograd.grad(
                h,
                x,
                torch.ones_like(h),
                create_graph=True,
                only_inputs=True,
            )[0]
        return grad

    def prox(
        self,
        x,
        *args,
        gamma=1.0,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
        **kwargs,
    ):
        r"""
        Calculates the proximity operator of :math:`h` at :math:`x`. By default, the proximity operator is computed using internal gradient descent.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma h}(x)`, computed in :math:`x`.
        """
        grad = lambda z: gamma * self.grad(z, *args, **kwargs) + (z - x)
        return gradient_descent(
            grad, x, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )

    def prox_conjugate(self, x, *args, gamma=1.0, lamb=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the convex conjugate :math:`(\lambda h)^*` at :math:`x`, using the Moreau formula.

        ::Warning:: Only valid for convex potential.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param float lamb: math:`\lambda` parameter in front of :math:`f`
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma \lambda h)^*}(x)`, computed in :math:`x`.
        """
        return x - gamma * self.prox(x / gamma, *args, gamma=lamb / gamma, **kwargs)

    def bregman_prox(
        self,
        x,
        bregman_potential,
        *args,
        gamma=1.0,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
        **kwargs,
    ):
        r"""
        Calculates the (right) Bregman proximity operator of h` at :math:`x`, with Bregman potential `bregman_potential`.

        .. math::

            \operatorname{prox}^h_{\gamma \regname}(x) = \underset{u}{\text{argmin}} \frac{\gamma}{2}h(u) + D_\phi(u,x)

        where :math:`D_\phi(x,y)` stands for the Bregman divergence with potential :math:`\phi`.

        By default, the proximity operator is computed using internal gradient descent.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param deepinv.optim.bregman.Bregman bregman_potential: Bregman potential to be used in the Bregman proximity operator.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}^h_{\gamma \regname}(x)`, computed in :math:`x`.
        """
        grad = lambda u: gamma * self.grad(u, *args, **kwargs) + (
            bregman_potential.grad(u) - bregman_potential.grad(x)
        )
        return gradient_descent(
            grad, x, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )
