import torch

from .optim_iterator import OptimIterator, fStep, gStep


class DRSIteration(OptimIterator):
    r"""
    Iterator for Douglas-Rachford Splitting.

    Class for a single iteration of the Douglas-Rachford Splitting (DRS) algorithm for minimizing
    :math:`f(x) + \lambda \regname(x)`.

    If the attribute ``g_first`` is set to False (by default), the iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\gamma f}(z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(2*u_{k+1}-z_k) \\
        z_{k+1} &= z_k + \beta (x_{k+1} - u_{k+1})
        \end{aligned}
        \end{equation*}

    where :math:`\gamma>0` is a stepsize and :math:`\beta>0` is a relaxation parameter.

    If the attribute ``g_first`` is set to True, the functions :math:`f` and :math:`\regname` are inverted in the previous iteration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.g_step = gStepDRS(**kwargs)
        self.f_step = fStepDRS(**kwargs)
        self.requires_prox_g = True

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        Single iteration of the DRS algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x, z = X["est"]
        if z.shape != x.shape:
            # In DRS, the "dual" variable z is a fake dual variable as it lives in the primal, hence this line to prevent from usual initialisation
            z = torch.zeros_like(x)
        if self.g_first:
            u = self.g_step(x, z, cur_prior, cur_params)
            x = self.f_step(u, z, cur_data_fidelity, cur_params, y, physics)
        else:
            u = self.f_step(x, z, cur_data_fidelity, cur_params, y, physics)
            x = self.g_step(u, z, cur_prior, cur_params)
        z = z + cur_params["beta"] * (x - u)
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"est": (x, z), "cost": F}


class fStepDRS(fStep):
    r"""
    DRS fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepDRS, self).__init__(**kwargs)

    def forward(self, x, z, cur_data_fidelity, cur_params, y, physics):
        r"""
        Single iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current first variable.
        :param torch.Tensor z: Current second variable.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the data-fidelity term.
        """
        if self.g_first:
            p = 2 * x - z
        else:
            p = z
        return cur_data_fidelity.prox(p, y, physics, gamma=cur_params["stepsize"])


class gStepDRS(gStep):
    r"""
    DRS gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepDRS, self).__init__(**kwargs)

    def forward(self, x, z, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`\lambda g`.

        :param torch.Tensor x:  Current first variable.
        :param torch.Tensor z: Current second variable.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        if self.g_first:
            p = z
        else:
            p = 2 * x - z
        return cur_prior.prox(
            p,
            cur_params["g_param"],
            gamma=cur_params["lambda"] * cur_params["stepsize"],
        )
