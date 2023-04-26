import torch
from torch import nn

from .denoiser import register


@register("prox_l1")
class ProxL1Prior(nn.Module):
    r"""
    Proximity operator of the l1 norm.


    This model is defined as the solution to the optimization problem:

    .. math::

        \underset{x}{\arg\min} \;  \|x-y\|^2 + \lambda \|x\|_1

    where :math:`\lambda>0` is a hyperparameter.

    The solution is available in closed-form as the soft-thresholding operator.
    """

    def __init__(self):
        super(ProxL1Prior, self).__init__()

    def prox_l1(self, x, ths=0.1):
        return torch.maximum(
            torch.tensor([0], device=x.device).type(x.dtype), x - ths
        ) + torch.minimum(torch.tensor([0], device=x.device).type(x.dtype), x + ths)

    def forward(self, x, sigma):
        return self.prox_l1(x, sigma)
