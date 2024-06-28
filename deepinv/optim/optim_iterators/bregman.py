import torch
import torch.nn as nn


class Bregman(nn.Module):
    r"""
    Module for the Bregman framework with convex Bregman potential :math:`h`.
    """

    def __init__(self, h=None):
        super().__init__()
        self._h = h

    def h(self, x, *args, **kwargs):
        r"""
        Computes the potential :math:`h(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) prior :math:`h(x)`.
        """
        return self._h(x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the potential :math:`h(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`h(x)`.
        """
        return self.h(x, *args, **kwargs)

    def conjugate(self, x, *args, **kwargs):
        r"""
        Computes the convex conjugate potential :math:`h^*(y) = \sup_{x} \langle x, y \rangle - h(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`h^*(y)`.
        """
        pass

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the potential :math:`h` at :math:`x`.
        By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h`, computed in :math:`x`.
        """
        with torch.enable_grad():
            x = x.requires_grad_()
            grad = torch.autograd.grad(
                self.h(x, *args, **kwargs), x, create_graph=True, only_inputs=True
            )[0]
        return grad

    def grad_conj(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the convex conjugate :math:`h^*` of :math:`h`.
        It corresponds to the inverse of the gradient of :math:`h`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h^*`, computed in :math:`x`.
        """
        pass

    def div(self, x, y, *args, **kwargs):
        r"""
        Computes the Bregman divergence :math:`D_h(x,y)` with potential :math:`h`.

        :param torch.Tensor x: Left variable :math:`x` at which the divergence is computed.
        :param torch.Tensor y: Right variable :math:`y` at which the divergence is computed.

        :return: (torch.tensor) divergence :math:`h(x) - h(y) - \langle \nabla h(y), x-y`.
        """
        return (
            self.h(x, *args, **kwargs)
            - self.h(y, *args, **kwargs)
            - torch.sum(
                (self.grad(y, *args, **kwargs) * (x - y)).reshape(x.shape[0], -1),
                dim=-1,
            )
        )

    def MD_step(self, x, grad, *args, gamma=1.0, **kwargs):
        r"""
        Performs a Mirror Descent step :math:`x = \nabla h^*(\nabla h(x) - \gamma \nabla f(x))`.

        :param torch.Tensor x: Variable :math:`x` at which the step is performed.
        :param torch.Tensor grad: Gradient of the minimized function at :math:`x`.
        :param float gamma: Step size.
        :return: (torch.tensor) updated variable :math:`x`.
        """
        return self.grad_conj(self.grad(x, *args, **kwargs) - gamma * grad)


class BregmanL2(Bregman):
    r"""
    Module for the L2 norm as Bregman potential :math:`h(x) = \frac{1}{2} \|x\|_2^2`.
    The corresponding Bregman divergence is the squared Euclidean distance :math:`D(x,y) = \frac{1}{2} \|x-y\|_2^2`.
    """

    def __init__(self):
        super().__init__()

    def h(self, x):
        r"""
        Computes the L2 norm potential :math:`h(x) = \frac{1}{2} \|x\|_2^2`.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`h(x)`.
        """
        return 0.5 * torch.sum(x.reshape(x.shape[0], -1) ** 2, dim=-1)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential :math:`h^*(y) = \frac{1}{2} \|y\|_2^2`.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`h^*(y)`.
        """
        return 0.5 * torch.sum(x.reshape(x.shape[0], -1) ** 2, dim=-1)

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the L2 norm :math:`\nabla h(x) = x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h`, computed in :math:`x`.
        """
        return x

    def grad_conj(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the conjugate of the L2 norm :math:`\nabla h^*(x) = x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h^*`, computed in :math:`x`.
        """
        return x

    def div(self, x, y, *args, **kwargs):
        r"""
        Computes the Bregman divergence with potential :math:`h`.

        :param torch.Tensor x: Variable :math:`x` at which the divergence is computed.
        :param torch.Tensor y: Variable :math:`y` at which the divergence is computed.

        :return: (torch.tensor) divergence :math:`h(x) - h(y) - \langle \nabla h(y), x-y`.
        """
        return 0.5 * torch.sum((x - y).reshape(x.shape[0], -1) ** 2, dim=-1)


class BurgEntropy(Bregman):
    r"""
    Module for the using Burg's entropy as Bregman potential :math:`h(x) = - \sum_i \log x_i`.
    The corresponding Bregman divergence is the Itakura-Saito distance :math:`D(x,y) = \sum_i x_i / y_i - \log(x_i / y_i) - 1`.
    """

    def __init__(self):
        super().__init__()

    def h(self, x):
        r"""
        Computes Burg's entropy potential :math:`h(x) = - \sum_i \log x_i`.
        The input :math:`x` must be postive.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`h(x)`.
        """
        return -torch.sum(torch.log(x).reshape(x.shape[0], -1), dim=-1)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential :math:`h^*(y) = - - \sum_i \log (-x_i)`.
        The input :math:`x` must be negative.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`h^*(y)`.
        """
        n = torch.shape(x.reshape(x.shape[0], -1))[-1]
        return -torch.sum(torch.log(-x).reshape(x.shape[0], -1), dim=-1) - n

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of Burg's entropy :math:`\nabla h(x) = - 1 / x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h`, computed in :math:`x`.
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
    Module for the using negative entropy as Bregman potential :math:`h(x) = \sum_i x_i \log x_i`.
    The corresponding Bregman divergence is the Kullback-Leibler divergence :math:`D(x,y) = \sum_i x_i \log(x_i / y_i) - x_i + y_i`.
    """

    def __init__(self):
        super().__init__()

    def h(self, x):
        r"""
        Computes negative entropy potential :math:`h(x) = \sum_i x_i \log x_i`.
        The input :math:`x` must be postive.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`h(x)`.
        """
        return torch.sum((x * torch.log(x)).reshape(x.shape[0], -1), dim=-1)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential :math:`h^*(y) = \sum_i y_i \log y_i`.
        The input :math:`x` must be negative.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`h^*(y)`.
        """
        return torch.sum(torch.exp(x - 1).reshape(x.shape[0], -1), dim=-1)

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of negative entropy :math:`\nabla h(x) = 1 + \log x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h`, computed in :math:`x`.
        """
        return 1 + torch.log(x)

    def grad_conj(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the conjugate of negative entropy :math:`\nabla h^*(x) = 1 + \log x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h^*`, computed in :math:`x`.
        """
        return torch.exp(x - 1)
