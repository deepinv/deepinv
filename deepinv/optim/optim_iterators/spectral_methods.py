import torch

from .optim_iterator import OptimIterator, fStep, gStep


class SMIteration(OptimIterator):
    r"""
    Iterator for Spectral Methods for :class:`deepinv.physics.PhaseRetrieval`.
    
    Class for a single iteration of the Spectral Methods algorithm to find the principal eigenvector of the regularized weighted covariance matrix:
    
    .. math::
        \begin{equation*}
        M = \conj{B} \text{diag}(T(y)) B + \lambda I,
        \end{equation*}
    
    where :math:`B` is the linear operator of the phase retrieval class, :math:`T(\cdot)` is a preprocessing function for the measurements, and :math:`I` is the identity matrix of corresponding dimensions. Parameter :math:`\lambda` tunes the strength of regularization.

    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        x_{k+1} &= M x_k \\
        x_{k+1} &= \operatorname{prox}_{\gamma g}(x_{k+1}),
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` is a stepsize that should satisfy :math:`\lambda \gamma \leq 2/\operatorname{Lip}(\|\nabla f\|)`.
    """

    def __init__(
        self,
        lamb=10,
        n_iter=50,
        preprocessing=lambda x: torch.max(1 - 1 / x, torch.tensor(-5.0)),
        **kwargs,
    ):
        super(SMIteration, self).__init__()
        self.n_iter = n_iter
        self.f_step = fStepSM(lamb, preprocessing=preprocessing, **kwargs)
        self.g_step = gStepSM(**kwargs)

    def forward(self, x, cur_prior, cur_params, y, physics, *args):
        r"""
        Single iteration of the spectral method.

        :param dict x: the current iterate :math:`x_k`.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics containing the forward operator.
        :return: The new iterate :math:`x_{k+1}`.
        """
        assert hasattr(
            physics, "B"
        ), "The physics should inherit from the PhaseRetrieval class."
        assert hasattr(
            physics, "B_adjoint"
        ), "The physics should inherit from the PhaseRetrieval class."
        x = self.f_step(x, y, physics)
        x = self.g_step(x, cur_prior, cur_params)
        return x


class fStepSM(fStep):
    r"""
    Spectral Methods fStep module.
    """

    def __init__(
        self,
        lamb=10,
        preprocessing=lambda x: torch.max(1 - 1 / x, torch.tensor(-5.0)),
        **kwargs,
    ):
        super(fStepSM, self).__init__(**kwargs)
        self.preprocessing = preprocessing
        self.lamb = lamb

    def forward(self, x: torch.Tensor, y: torch.Tensor, physics):
        r"""
        Single power iteration step for spectral methods.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the forward matrix.
        """
        x = x.to(torch.cfloat)
        # normalize every image in x
        x = torch.stack([subtensor / subtensor.norm() for subtensor in x])
        # y should have mean 1 for each image
        y = y / torch.mean(y, dim=1, keepdim=True)
        diag_T = self.preprocessing(y)
        diag_T = diag_T.to(torch.cfloat)
        res = physics.B(x)
        res = diag_T * res
        res = physics.B_adjoint(res)
        x = res + self.lamb * x
        x = torch.stack([subtensor / subtensor.norm() for subtensor in x])
        return x


class gStepSM(gStep):
    r"""
    Spectral Methods gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepSM, self).__init__(**kwargs)

    def forward(self, x: torch.Tensor, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        return cur_prior.prox(
            x,
            cur_params["g_param"],
            gamma=cur_params["lambda"] * cur_params["stepsize"],
        )
