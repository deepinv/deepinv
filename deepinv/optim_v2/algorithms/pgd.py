import torch

from deepinv.models import Reconstructor
from deepinv.optim import DataFidelity, Prior
from deepinv.physics import Physics


class PGD(Reconstructor):
    r"""
    Proximal gradient descent for minimizing :math:`f(x) + \lambda g(x)`.

    Each iteration applies

    .. math::

        x_{k+1} = \operatorname{prox}_{\gamma\lambda g}
        (x_k - \gamma\nabla f(x_k)).

    :param deepinv.optim.DataFidelity data_fidelity: data-fidelity term :math:`f`.
    :param deepinv.optim.Prior prior: prior :math:`g`.
    :param float stepsize: gradient stepsize :math:`\gamma`. Default: ``1.0``.
    :param float lambda_reg: prior weight :math:`\lambda`. Default: ``1.0``.
    :param int max_iter: number of iterations. Default: ``100``.
    """

    def __init__(
        self,
        data_fidelity: DataFidelity,
        prior: Prior,
        stepsize: float = 1.0,
        lambda_reg: float = 1.0,
        max_iter: int = 100,
    ):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter

    def forward(
        self, y: torch.Tensor, physics: Physics, init: torch.Tensor = None
    ) -> torch.Tensor:
        r"""
        Run proximal gradient descent from ``init`` or :math:`A^\top y`.

        :param torch.Tensor y: measurements.
        :param deepinv.physics.Physics physics: forward model.
        :param torch.Tensor init: initial iterate. Default: ``None``.
        :return: reconstructed signal.
        """
        x = physics.A_adjoint(y) if init is None else init
        for _ in range(self.max_iter):
            x = x - self.stepsize * self.data_fidelity.grad(x, y, physics)
            x = self.prior.prox(x, gamma=self.lambda_reg * self.stepsize)
        return x
