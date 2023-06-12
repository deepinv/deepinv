import torch

from .optim_iterator import OptimIterator, fStep, gStep


class DRSIteration(OptimIterator):
    r"""
    Single iteration of DRS.

    Class for a single iteration of the Douglas-Rachford Splitting (DRS) algorithm for minimising
    :math:`\lambda f(x) + g(x)`.

    If the attribute `"g_first"`is set to False (by default), the iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\gamma \lambda f}(z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma g}(2*u_{k+1}-z_k) \\
        z_{k+1} &= z_k + \beta (x_{k+1} - u_{k+1})
        \end{aligned}
        \end{equation*}

    where :math:`\gamma>0` is a stepsize and :math:`\beta>0` is a relaxation parameter.

    If the attribute `"g_first"`is set to True, the functions :math:`f` and :math:`g` are inverted in the previous iteration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.g_step = gStepDRS(**kwargs)
        self.f_step = fStepDRS(**kwargs)
        self.requires_prox_g = True

    def forward(self, X, cur_prior, cur_params, y, physics):
        r"""
        Single iteration of the DRS algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: dictionary containing the current parameters of the model.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x, z = X["est"]
        if z.shape != x.shape:
            # In DRS, the "dual" variable z is a fake dual variable as it lives in the primal, hence this line to prevent from usual initialisation
            z = torch.zeros_like(x)
        if self.g_first:
            u = self.g_step(x, z, cur_prior, cur_params)
            x = self.f_step(u, z, y, physics, cur_params)
        else:
            u = self.f_step(x, z, y, physics, cur_params)
            x = self.g_step(u, z, cur_prior, cur_params)
        z = z + self.beta * (x - u)
        F = self.F_fn(x, cur_prior, cur_params, y, physics) if self.F_fn else None
        return {"est": (x, z), "cost": F}


class fStepDRS(fStep):
    r"""
    DRS fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepDRS, self).__init__(**kwargs)

    def forward(self, x, z, y, physics, cur_params):
        r"""
        Single iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :param dict cur_params: Dictionary containing the current fStep parameters (keys `"stepsize"` and `"lambda"`).
        """
        if self.g_first:
            p = 2 * x - z
        else:
            p = z
        return self.data_fidelity.prox(
            p, y, physics, cur_params["lambda"] * cur_params["stepsize"]
        )


class gStepDRS(gStep):
    r"""
    DRS gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepDRS, self).__init__(**kwargs)

    def forward(self, x, z, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`g`.

        :param torch.Tensor z: Current iterate :math:`z_k`.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current gStep parameters (keys `"prox_g"` and `"g_param"`).
        """
        if self.g_first:
            p = z
        else:
            p = 2 * x - z
        return cur_prior.prox(p, cur_params["stepsize"], cur_params["g_param"])
