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
        :return: (torch.tensor) gradient :math:`\nabla_x h`, computed in :math:`x`.
        """
        pass
    
    def div(self, x, y, *args, **kwargs):
        r"""
        Computes the Bregmab divergence with potential :math:`h`.

        :param torch.Tensor x: Variable :math:`x` at which the divergence is computed.
        :param torch.Tensor y: Variable :math:`y` at which the divergence is computed.

        :return: (torch.tensor) divergence :math:`h(x) - h(y) - \langle \nabla h(y), x-y`.
        """
        return self.h(x, *args, **kwargs) - self.h(y, *args, **kwargs) - torch.sum((self.grad(y, *args, **kwargs) * (x - y)).reshape(x.shape[0],-1), dim=-1)
    
    def MD_step(self, x, grad, gamma, *args, **kwargs):
        r"""
        Performs a Mirror Descent step :math:`x = \nabla h^*(\nabla h(x) - \gamma \nabla f(x))`.

        :param torch.Tensor x: Variable :math:`x` at which the step is performed.
        :param torch.Tensor grad: Gradient of the minimized function at :math:`x`.
        :param float gamma: Step size.
        :return: (torch.tensor) updated variable :math:`x`.
        """
        return self.grad_conj(self.grad(x, *args, **kwargs) - gamma*grad)



class BurgEntropy(Bregman):
    r"""
    Module for the using the Burg entropy as Bregman potential :math:`h(x) = \sum_i x_i \log x_i - x_i`.
    """

    def __init__(self):
        super().__init__()

    def h(self, x):
        r"""
        Computes the potential :math:`h(x) = \sum_i x_i \log x_i - x_i`.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`h(x)`.
        """
        return torch.sum(x * torch.log(x) - x)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential :math:`h^*(y) = \sup_{x} \langle x, y \rangle - h(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`h^*(y)`.
        """
        return torch.sum(torch.exp(x) - 1)

    def grad_conj(self, x):
        r"""
        Calculates the gradient of the convex conjugate :math:`h^*` of :math:`h`. 
        It corresponds to the inverse of the gradient of :math:`h`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h`, computed in :math:`x`.
        """
        return torch.exp(x)

    def grad(self, x):
        r"""
        Calculates the gradient of the potential :math:`h` at :math:`x`.
        By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x h`, computed in :math:`x`.
        """
        return torch.log(x) + 1

    def div(self, x, y):
        r"""
        Computes the Bregmab divergence with potential :math:`h`.

        :param torch.Tensor x: Variable :math:`x` at which the divergence is computed.
        :param torch.Tensor y: Variable :math:`y` at which the divergence is computed