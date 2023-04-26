from .optim_iterator import OptimIterator, fStep, gStep


class DRSIteration(OptimIterator):
    r"""
    Single iteration of DRS.

    Class for a single iteration of the Douglas-Rachford Splitting (DRS) algorithm for minimising
    :math:`\lambda f(x) + g(x)`.

    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= \operatorname{prox}_{f}(x_k) \\
        x_{k+1/2} &= \operatorname{prox}_{g}(u_k) \\
        x_{k+1} &= (x_{k+1/2} - x_{k})/2
        \end{aligned}
        \end{equation*}
    """

    def __init__(self, **kwargs):
        super(DRSIteration, self).__init__(**kwargs)
        self.g_step = gStepDRS(**kwargs)
        self.f_step = fStepDRS(**kwargs)

    def forward(self, X, cur_prior, cur_params, y, physics):
        r"""
        Single iteration of the DRS algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param dict cur_prior: dictionary containing the prior-related term of interest, e.g. its proximal operator or gradient.
        :param dict cur_params: dictionary containing the current parameters of the model.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"][0]
        if not self.g_first:
            x = self.f_step(x_prev, y, physics, cur_params)
            x = self.g_step(x, cur_prior, cur_params)
        else:
            x = self.g_step(x_prev, cur_prior, cur_params)
            x = self.f_step(x, y, physics, cur_params)
        x = (x_prev + x) / 2.0
        x = self.relaxation_step(x, x_prev)
        F = self.F_fn(x, cur_params, y, physics) if self.F_fn else None
        return {"est": (x,), "cost": F}


class fStepDRS(fStep):
    r"""
    DRS fStep module
    """

    def __init__(self, **kwargs):
        super(fStepDRS, self).__init__(**kwargs)

    def forward(self, x, y, physics, cur_params):
        r"""
        Single iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :param dict cur_params: Dictionary containing the current fStep parameters (keys `"stepsize"` and `"lambda"`).
        """
        return (
            2
            * self.data_fidelity.prox(
                x, y, physics, 1 / (cur_params["lambda"] * cur_params["stepsize"])
            )
            - x
        )


class gStepDRS(gStep):
    r"""
    DRS gStep module
    """

    def __init__(self, **kwargs):
        super(gStepDRS, self).__init__(**kwargs)

    def forward(self, z, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`g`.

        :param torch.Tensor z: Current iterate :math:`z_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current gStep parameters (keys `"prox_g"` and `"g_param"`).
        """
        return 2 * cur_prior["prox_g"](z, cur_params["g_param"]) - z
