import torch
from .optim_iterator import OptimIterator, fStep, gStep


class ADMMIteration(OptimIterator):
    r"""
    Single iteration of ADMM.

    Class for a single iteration of the Alternating Direction Method of Multipliers (ADMM) algorithm for
    minimising :math:`\lambda f(x) + g(x)`.

    If the attribute ``g_first`` is set to False (by default),
    the iteration is (`see this paper <https://www.nowpublishers.com/article/Details/MAL-016>`_):

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\gamma \lambda f}(x_k - z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma g}(u_{k+1} + z_k) \\
        z_{k+1} &= z_k + \beta (u_{k+1} - x_{k+1})
        \end{aligned}
        \end{equation*}

    where :math:`\gamma>0` is a stepsize and :math:`\beta>0` is a relaxation parameter.
    Here, the concatenation :math:`(x_k,z_k)` is the iterate i.e. the fixed point variable iterated by the algorithm and :math:`x_k` is the estimate i.e. the estimation of the solution of the minimization problem.

    If the attribute ``g_first`` is set to ``True``, the functions :math:`f` and :math:`g` are
    inverted in the previous iteration.

    """

    def __init__(self, **kwargs):
        super(ADMMIteration, self).__init__(**kwargs)
        self.g_step = gStepADMM(**kwargs)
        self.f_step = fStepADMM(**kwargs)
        self.requires_prox_g = True

    def get_minimizer_from_FP(self, x, cur_data_fidelity, cur_prior, cur_params, y, physics):
        """
        Get the minimizer of F from the fixed point variable x.

        :param torch.Tensor x: Fixed point variable iterated by the algorithm.
        :return: Minimizer of F.
        """
        return x[0]

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        Single iteration of the ADMM algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x,z = X["fp"]
        if z.shape != x.shape:
            # In ADMM, the "dual" variable z is a fake dual variable as it lives in the primal, hence this line to prevent from usual initialisation
            z = torch.zeros_like(x)
        if self.g_first:
            u = self.g_step(x, z, cur_prior, cur_params)
            x = self.f_step(u, z, cur_data_fidelity, cur_params, y, physics)
        else:
            u = self.f_step(x, z, cur_data_fidelity, cur_params, y, physics)
            x = self.g_step(u, z, cur_prior, cur_params)
        z = z + cur_params["beta"] * (u - x)
        fp = (x,z)
        est = self.get_minimizer_from_FP(fp, cur_data_fidelity, cur_prior, cur_params, y, physics)
        F = (
            self.F_fn(est, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"fp" : fp, "est": est, "cost": F}


class fStepADMM(fStep):
    r"""
    ADMM fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepADMM, self).__init__(**kwargs)

    def forward(self, x, z, cur_data_fidelity, cur_params, y, physics):
        r"""
        Single iteration step on the data-fidelity term :math:`\lambda f`.

        :param torch.Tensor x: current first variable
        :param torch.Tensor z: current second variable
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the observation.
        """
        if self.g_first:
            p = x + z
        else:
            p = x - z
        return cur_data_fidelity.prox(
            p, y, physics, cur_params["lambda"] * cur_params["stepsize"]
        )


class gStepADMM(gStep):
    r"""
    ADMM gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepADMM, self).__init__(**kwargs)

    def forward(self, x, z, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`g`.

        :param torch.Tensor x: current first variable
        :param torch.Tensor z: current second variable
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        if self.g_first:
            p = x - z
        else:
            p = x + z
        return cur_prior.prox(p, cur_params["stepsize"], cur_params["g_param"])
