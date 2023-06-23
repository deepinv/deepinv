from .optim_iterator import OptimIterator, fStep, gStep


class HQSIteration(OptimIterator):
    r"""
    Single iteration of HQS.

    Class for a single iteration of the Half-Quadratic Splitting (HQS) algorithm for minimising :math:`\lambda f(x) + g(x)`.
    The iteration is given by


    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= \operatorname{prox}_{\gamma \lambda f}(x_k) \\
        x_{k+1} &= \operatorname{prox}_{\sigma g}(u_k).
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` and :math:`\sigma` are step-sizes. Note that this algorithm does not converge to
    a minimizer of :math:`\lambda f(x) + g(x)`, but instead to a minimizer of
    :math:`\lambda \gamma\, ^1f+\sigma g`, where :math:`^1f` denotes
    the Moreau envelope of :math:`f`

    """

    def __init__(self, **kwargs):
        super(HQSIteration, self).__init__(**kwargs)
        self.g_step = gStepHQS(**kwargs)
        self.f_step = fStepHQS(**kwargs)
        self.requires_prox_g = True


class fStepHQS(fStep):
    r"""
    HQS fStep module.
    """

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
            x, y, physics, cur_params["lambda"] * cur_params["stepsize"]
        )


class gStepHQS(gStep):
    r"""
    HQS gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepHQS, self).__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param dict cur_prior: Class containing the current prior.
        :param dict cur_params: Dictionary containing the current gStep parameters (keys `"prox_g"` and `"g_param"`).
        """
        return cur_prior.prox(x, cur_params["stepsize"], cur_params["g_param"])
