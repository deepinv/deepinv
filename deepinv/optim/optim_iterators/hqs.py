from .optim_iterator import OptimIterator, fStep, gStep


class HQSIteration(OptimIterator):
    r"""
    Single iteration of HQS.

    Class for a single iteration of the Half-Quadratic Splitting (HQS) algorithm for minimising :math:`\lambda f(x) + g(x)`.
    The iteration is given by


    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= \operatorname{prox}_{f}(x_k) \\
        x_{k+1} &= \operatorname{prox}_{g}(u_k)
        \end{aligned}
        \end{equation*}


    """
    def __init__(self, **kwargs):
        super(HQSIteration, self).__init__(**kwargs)
        self.g_step = gStepHQS(**kwargs)
        self.f_step = fStepHQS(**kwargs)


class fStepHQS(fStep):
    def __init__(self, **kwargs):
        super(fStepHQS, self).__init__(**kwargs)

    def forward(self, x, cur_params, y, physics):
        r"""
        Single iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param dict cur_params: Dictionary containing the current fStep parameters (keys `"stepsize"` and `"lambda"`).
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        """
        return self.data_fidelity.prox(
            x, y, physics, 1 / (cur_params["lambda"] * cur_params["stepsize"])
        )


class gStepHQS(gStep):
    def __init__(self, **kwargs):
        super(gStepHQS, self).__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current gStep parameters (keys `"prox_g"` and `"g_param"`).
        """
        return cur_prior["prox_g"](x, cur_params["g_param"])
