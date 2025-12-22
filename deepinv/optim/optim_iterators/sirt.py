from __future__ import annotations
from .optim_iterator import OptimIterator, fStep, gStep
from deepinv.optim.bregman import Bregman, BregmanL2


class SIRTIteration(OptimIterator):
    r"""
    Iterator for Simultaneous Iterative Reconstruction Technique (SIRT).

    Class for a single iteration of the SIRT algorithm for minimising :math:`f(x)`.

    The iteration is given by


    .. math::
        \begin{equation*}
        x_{k+1} = x_k + \gamma V A^\top W (A x_k - y)
        \end{equation*}


    where :
    - :math:`\gamma` is a stepsize.
    - :math:`W = \mathrm{diag}\left(\frac{1}{\sum_{i}a_{ij}}\right)`, a diagonal matrix where each diagonal element is the inverse of the sum of the elements of the corresponding row of the forward operator :math:`A`,
    - :math:`V = \mathrm{diag}\left(\frac{1}{\sum_{j}a_{ij}}\right)`, a diagonal matrix where each diagonal element is the inverse of the sum of the elements of the corresponding column of the forward operator :math:`A`.

    """

    def __init__(self, **kwargs):
        super(SIRTIteration, self).__init__(**kwargs)
        self.f_step = fStepSIRT(**kwargs)

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        Single SIRT iteration on the objective :math:`f(x)`.

        :param dict X: Dictionary containing the current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"][0]

        x = x_prev + cur_params["stepsize"] * self.f_step(
            x_prev, cur_data_fidelity, cur_params, y, physics
        )

        F = (
            self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            and self.cost_fn is not None
            and cur_data_fidelity is not None
            else None
        )
        return {"est": (x,), "cost": F}


class fStepSIRT(fStep):
    r"""
    Data fidelity step for SIRT.

    Class for the data fidelity step of the SIRT algorithm for minimising :math:`f(x)`.

    The step is given by


    .. math::
        \begin{equation*}
        fStep(x_k) = V A^\top W (A x_k - y)
        \end{equation*}


    where :
    - :math:`W = \mathrm{diag}\left(\frac{1}{\sum_{i}a_{ij}}\right)`, a diagonal matrix where each diagonal element is the inverse of the sum of the elements of the corresponding row of the forward operator :math:`A`,
    - :math:`V = \mathrm{diag}\left(\frac{1}{\sum_{j}a_{ij}}\right)`, a diagonal matrix where each diagonal element is the inverse of the sum of the elements of the corresponding column of the forward operator :math:`A`.

    """

    def forward(self, x, cur_data_fidelity, cur_params, y, physics, *args, **kwargs):
        r"""
        Data fidelity step of SIRT.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :return: Data fidelity step evaluated at :math:`x_k`.
        """
        A_x = physics.forward(x)
        residual = A_x - y
