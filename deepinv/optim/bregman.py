import torch
import torch.nn as nn
from deepinv.optim.potential import Potential


class Bregman(Potential):
    r"""
    Module for the Bregman framework with convex Bregman potential :math:`\phi`.
    Comes with methods to compute the potential, its gradient, its conjugate, its gradient and its Bregman divergence.

    :param Callable h: Potential function :math:`\phi(x)` to be used in the Bregman framework.
    """

    def __init__(self, phi=None):
        super().__init__(fn=phi)

    def div(self, x, y, *args, **kwargs):
        r"""
        Computes the Bregman divergence :math:`D_\phi(x,y)` with Bregman potential :math:`\phi`.

        :param torch.Tensor x: Left variable :math:`x` at which the divergence is computed.
        :param torch.Tensor y: Right variable :math:`y` at which the divergence is computed.

        :return: (torch.tensor) divergence :math:`h(x) - h(y) - \langle \nabla h(y), x-y  \rangle`.
        """
        return (
            self(x, *args, **kwargs)
            - self(y, *args, **kwargs)
            - torch.sum(
                (self.grad(y, *args, **kwargs) * (x - y)).reshape(x.shape[0], -1),
                dim=-1,
            )
        )

    def MD_step(self, x, grad, *args, gamma=1.0, **kwargs):
        r"""
        Performs a Mirror Descent step :math:`x = \nabla \phi^*(\nabla \phi(x) - \gamma \nabla f(x))`.

        :param torch.Tensor x: Variable :math:`x` at which the step is performed.
        :param torch.Tensor grad: Gradient of the minimized function at :math:`x`.
        :param float gamma: Step size.
        :return: (torch.tensor) updated variable :math:`x`.
        """
        return self.grad_conj(self.grad(x, *args, **kwargs) - gamma * grad)


class BregmanL2(Bregman):
    r"""
    Module for the L2 norm as Bregman potential :math:`\phi(x) = \frac{1}{2} \|x\|_2^2`.
    The corresponding Bregman divergence is the squared Euclidean distance :math:`D(x,y) = \frac{1}{2} \|x-y\|_2^2`.
    """

    def __init__(self):
        super().__init__()

    def fn(self, x):
        r"""
        Computes the L2 norm potential :math:`\phi(x) = \frac{1}{2} \|x\|_2^2`.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`h(x)`.
        """
        return 0.5 * torch.sum(x.reshape(x.shape[0], -1) ** 2, dim=-1)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential :math:`\phi^*(y) = \frac{1}{2} \|y\|_2^2`.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`\phi^*(y)`.
        """
        return 0.5 * torch.sum(x.reshape(x.shape[0], -1) ** 2, dim=-1)

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the L2 norm :math:`\nabla \phi(x) = x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x \phi`, computed in :math:`x`.
        """
        return x

    def grad_conj(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the conjugate of the L2 norm :math:`\nabla \phi^*(x) = x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x \phi^*`, computed in :math:`x`.
        """
        return x

    def div(self, x, y, *args, **kwargs):
        r"""
        Computes the Bregman divergence with potential :math:`\phi`. Here falls back to the L2 distance.

        :param torch.Tensor x: Variable :math:`x` at which the divergence is computed.
        :param torch.Tensor y: Variable :math:`y` at which the divergence is computed.

        :return: (torch.tensor) divergence :math:`\phi(x) - \phi(y) - \langle \nabla \phi(y), x-y`.
        """
        return 0.5 * torch.sum((x - y).reshape(x.shape[0], -1) ** 2, dim=-1)


class BurgEntropy(Bregman):
    r"""
    Module for the using Burg's entropy as Bregman potential :math:`\phi(x) = - \sum_i \log x_i`.

    The corresponding Bregman divergence is the Itakura-Saito distance :math:`D(x,y) = \sum_i x_i / y_i - \log(x_i / y_i) - 1`.
    As shown in https://publications.ut-capitole.fr/id/eprint/25852/1/25852.pdf, it is the Bregman potential to use for performing mirror descent on the Poisson likelihood :class:`deepinv.optim.data_fidelity.PoissonLikelihood`.
    """

    def __init__(self):
        super().__init__()

    def fn(self, x):
        r"""
        Computes Burg's entropy potential :math:`\phi(x) = - \sum_i \log x_i`.
        The input :math:`x` must be postive.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`h(x)`.
        """
        return -torch.sum(torch.log(x).reshape(x.shape[0], -1), dim=-1)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential :math:`\phi^*(y) = - - \sum_i \log (-x_i)`.
        The input :math:`x` must be negative.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`\phi^*(y)`.
        """
        n = torch.shape(x.reshape(x.shape[0], -1))[-1]
        return -torch.sum(torch.log(-x).reshape(x.shape[0], -1), dim=-1) - n

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of Burg's entropy :math:`\nabla \phi(x) = - 1 / x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x \phi`, computed in :math:`x`.
        """
        return -1 / x

    def grad_conj(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the conjugate of Burg's entropy :math:`\nabla h^*(x) = - 1 / x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h^*`, computed in :math:`x`.
        """
        return -1 / x


class NegEntropy(Bregman):
    r"""
    Module for the using negative entropy as Bregman potential :math:`\phi(x) = \sum_i x_i \log x_i`.

    The corresponding Bregman divergence is the Kullback-Leibler divergence :math:`D(x,y) = \sum_i x_i \log(x_i / y_i) - x_i + y_i`.
    """

    def __init__(self):
        super().__init__()

    def fn(self, x):
        r"""
        Computes negative entropy potential :math:`\phi(x) = \sum_i x_i \log x_i`.
        The input :math:`x` must be postive.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`\phi(x)`.
        """
        return torch.sum((x * torch.log(x)).reshape(x.shape[0], -1), dim=-1)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential :math:`\phi^*(y) = \sum_i y_i \log y_i`.
        The input :math:`x` must be negative.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`\phi^*(y)`.
        """
        return torch.sum(torch.exp(x - 1).reshape(x.shape[0], -1), dim=-1)

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of negative entropy :math:`\nabla \phi(x) = 1 + \log x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x \phi`, computed in :math:`x`.
        """
        return 1 + torch.log(x)

    def grad_conj(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the conjugate of negative entropy :math:`\nabla \phi^*(x) = 1 + \log x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x \phi^*`, computed in :math:`x`.
        """
        return torch.exp(x - 1)


class Bregman_ICNN(Bregman):
    r"""
    Module for the using a deep ICNN as Bregman potential.
    """

    def __init__(self, forw_model, conj_model=None):
        super().__init__()
        self.forw_model = forw_model
        self.conj_model = conj_model

    def fn(self, x):
        r"""
        Computes the Bregman potential.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`\phi(x)`.
        """
        return self.forw_model(x)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`\phi^*(y)`.
        """
        if self.conj_model is not None:
            return self.conj_model(x)
        else:
            super().conjugate(x)
        return
