import torch
from .optim_iterator import OptimIterator, fStep, gStep


class ADMMIteration(OptimIterator):
    r"""
    Single iteration of ADMM.

    Class for a single iteration of the Alternating Direction Method of Multipliers (ADMM) algorithm for minimising :math:`\lambda f(x) + g(x)`.

    The iteration is given by `[ref]<https://www.nowpublishers.com/article/Details/MAL-016>`_:

    .. math::
        \begin{equation*}
        \begin{aligned}
        z_{k+1/2} &= \operatorname{prox}_{\gamma g}(x_k - z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma \lambda f}(z_{k+1/2} + z_k) \\
        z_{k+1} &= z_k + \beta (z_{k+1/2} - x_{k+1})
        \end{aligned}
        \end{equation*}


    where :math:`\gamma>0` is a stepsize.
    """

    def __init__(self, **kwargs):
        super(ADMMIteration, self).__init__(**kwargs)
        self.g_step = gStepADMM(**kwargs)
        self.f_step = fStepADMM(**kwargs)
        self.requires_prox_g = True

    def forward(self, X, cur_prior, cur_params, y, physics):
        r"""
        Single iteration of the ADMM algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param dict cur_prior: dictionary containing the prior-related term of interest, e.g. its proximal operator or gradient.
        :param dict cur_params: dictionary containing the current parameters of the model.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x, z = X["est"]

        if (
            z.shape != x.shape
        ):  # In ADMM, the "dual" variable u is a fake dual variable as it lives in the primal, hence this line to prevent from usual initialisation
            z = torch.zeros_like(x)

        z_prev = z.clone()

        z_temp = self.g_step(x, z, cur_prior, cur_params)
        x = self.f_step(z_temp, z, y, physics, cur_params)
        z = z_prev + self.beta * (z_temp - x)

        F = self.F_fn(x, cur_params, y, physics) if self.F_fn else None
        return {"est": (x, z), "cost": F}


class fStepADMM(fStep):
    r"""
    ADMM fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepADMM, self).__init__(**kwargs)

    def forward(self, x, u, y, physics, cur_params):
        r"""
        Single iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param torch.Tensor u: Current iterate :math:`u_k`.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :param dict cur_params: Dictionary containing the current fStep parameters (keys `"stepsize"` and `"lambda"`).
        """
        return self.data_fidelity.prox(
            x + u, y, physics, cur_params["lambda"] * cur_params["stepsize"]
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

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param torch.Tensor z: Current iterate :math:`z_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current gStep parameters (keys `"prox_g"` and `"g_param"`).
        """
        return cur_prior["prox_g"](x - z, cur_params["g_param"])
