from .optim_iterator import OptimIterator, fStep, gStep


class CPIteration(OptimIterator):
    r"""
    Single iteration of the Chambolle-Pock algorithm.

    Class for a single iteration of the `Chambolle-Pock <https://hal.science/hal-00490826/document>`_ Primal-Dual (PD)
    algorithm for minimising :math:`\lambda \datafid{Ax}{y} + g(x)`. Our implementation corresponds to
    Algorithm 1 of `<https://hal.science/hal-00609728v4/document>`_.

    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        x_{k+1} &= \operatorname{prox}_{\tau g}(x_k-\tau A^\top u_k) \\
        z_k &= 2Ax_{k+1}-x_k\\
        u_{k+1} &= \operatorname{prox}_{\sigma (\lambda f)^*}(z_k) \\
        \end{aligned}
        \end{equation*}

    where :math:`(\lambda f)^*` is the Fenchel-Legendre conjugate of :math:`\lambda f`, and :math:`\sigma` and :math:`\tau` are step-sizes that should
    satisfy :math:`\sigma \tau \|A\|^2 \leq 1`.
    """

    def __init__(self, **kwargs):
        super(CPIteration, self).__init__(**kwargs)
        self.g_step = gStepCP(**kwargs)
        self.f_step = fStepCP(**kwargs)

    def forward(self, X, cur_prior, cur_params, y, physics):
        r"""
        Single iteration of the Chambolle-Pock algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param dict cur_prior: dictionary containing the prior-related term of interest, e.g. its proximal operator or gradient.
        :param dict cur_params: dictionary containing the current parameters of the model.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev, u_prev = X["est"]

        x = self.g_step(x_prev, physics.A_adjoint(u_prev), cur_prior, cur_params)
        u = self.f_step(physics.A(2 * x - x_prev), u_prev, y, cur_params)

        F = self.F_fn(x, cur_prior, cur_params, y, physics) if self.F_fn else None

        return {"est": (x, u), "cost": F}


class fStepCP(fStep):
    r"""
    Chambolle-Pock fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepCP, self).__init__(**kwargs)

    def forward(self, Ax_cur, u, y, cur_params):
        r"""
        Single Chambolle-Pock iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor Ax_cur: Current iterate :math:`2Ax_{k+1}-x_k`
        :param torch.Tensor u: Current iterate :math:`u_k`.
        :param torch.Tensor y: Input data.
        :param dict cur_params: Dictionary containing the current fStep parameters (keys `"stepsize"` and `"lambda"`).
        """
        v = u + cur_params["stepsize"] * Ax_cur
        return v - cur_params["stepsize"] * self.data_fidelity.prox_d(
            v / cur_params["stepsize"], y, cur_params["lambda"] / cur_params["stepsize"]
        )


class gStepCP(gStep):
    r"""
    Chambolle-Pock gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepCP, self).__init__(**kwargs)

    def forward(self, x, Atu, cur_prior, cur_params):
        r"""
        Single Chambolle-Pock iteration step on the prior term :math:`g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param torch.Tensor Atu: Current iterate :math:`A^\top u_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current gStep parameters (keys `"prox_g"`, `"stepsize"` and `"g_param"`).
        """
        return cur_prior.prox(x - cur_params["g_param"] * Atu, cur_params["g_param"])
