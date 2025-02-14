import torch

from .optim_iterator import OptimIterator, fStep, gStep


class CPIteration(OptimIterator):
    r"""
    Iterator for Chambolle-Pock.

    Class for a single iteration of the `Chambolle-Pock <https://hal.science/hal-00490826/document>`_ Primal-Dual (PD)
    algorithm for minimising :math:`F(Kx) + \lambda G(x)` or :math:`\lambda F(x) + G(Kx)` for generic functions :math:`F` and :math:`G`.
    Our implementation corresponds to Algorithm 1 of `<https://hal.science/hal-00490826/document>`_.

    If the attribute ``g_first`` is set to ``False`` (by default), the iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\sigma F^*}(u_k + \sigma K z_k) \\
        x_{k+1} &= \operatorname{prox}_{\tau \lambda G}(x_k-\tau K^\top u_{k+1}) \\
        z_{k+1} &= x_{k+1} + \beta(x_{k+1}-x_k) \\
        \end{aligned}
        \end{equation*}

    where :math:`F^*` is the Fenchel-Legendre conjugate of :math:`F`, :math:`\beta>0` is a relaxation parameter, and :math:`\sigma` and :math:`\tau` are step-sizes that should
    satisfy :math:`\sigma \tau \|K\|^2 \leq 1`.

    If the attribute ``g_first`` is set to ``True``, the functions :math:`F` and :math:`G` are inverted in the previous iteration.

    In particular, setting :math:`F = \distancename`, :math:`K = A` and :math:`G = \regname`, the above algorithms solves

    .. math::

        \begin{equation*}
        \underset{x}{\operatorname{min}} \,\,  \distancename(Ax, y) + \lambda \regname(x)
        \end{equation*}


    with a splitting on :math:`\distancename`, with not differentiability assumption needed on :math:`\distancename`
    or :math:`\regname`, not any invertibility assumption on :math:`A`.

    Note that the algorithm requires an intiliazation of the three variables :math:`x_0`, :math:`z_0` and :math:`u_0`.
    """

    def __init__(self, **kwargs):
        super(CPIteration, self).__init__(**kwargs)
        self.g_step = gStepCP(**kwargs)
        self.f_step = fStepCP(**kwargs)

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        Single iteration of the Chambolle-Pock algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the data-fidelity term.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev, z_prev, u_prev = X["est"]  # x : primal, z : relaxed primal, u : dual
        K = lambda x: cur_params["K"](x) if "K" in cur_params.keys() else x
        K_adjoint = lambda x: (
            cur_params["K_adjoint"](x) if "K_adjoint" in cur_params.keys() else x
        )
        if self.g_first:
            u = self.g_step(u_prev, K(z_prev), cur_prior, cur_params)
            x = self.f_step(
                x_prev, K_adjoint(u), cur_data_fidelity, y, physics, cur_params
            )
        else:
            u = self.f_step(
                u_prev, K(z_prev), cur_data_fidelity, y, physics, cur_params
            )
            x = self.g_step(x_prev, K_adjoint(u), cur_prior, cur_params)
        z = x + cur_params["beta"] * (x - x_prev)
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"est": (x, z, u), "cost": F}


class fStepCP(fStep):
    r"""
    Chambolle-Pock fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepCP, self).__init__(**kwargs)

    def forward(self, x, w, cur_data_fidelity, y, physics, cur_params):
        r"""
        Single Chambolle-Pock iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current first variable :math:`x` if `"g_first"` and :math:`u` otherwise.
        :param torch.Tensor w: Current second variable :math:`A^\top u` if `"g_first"` and :math:`A z` otherwise.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the data-fidelity term.
        :param dict cur_params: Dictionary containing the current fStep parameters (keys `"stepsize_dual"` (or `"stepsize"`) and `"lambda"`).
        """
        if self.g_first:
            p = x - cur_params["stepsize"] * w
            return cur_data_fidelity.prox(p, y, physics, gamma=cur_params["stepsize"])
        else:
            p = x + cur_params["stepsize_dual"] * w
            return cur_data_fidelity.prox_conjugate(
                p, y, physics, gamma=cur_params["stepsize_dual"]
            )


class gStepCP(gStep):
    r"""
    Chambolle-Pock gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepCP, self).__init__(**kwargs)

    def forward(self, x, w, cur_prior, cur_params):
        r"""
        Single Chambolle-Pock iteration step on the prior term :math:`\lambda g`.

        :param torch.Tensor x: Current first variable :math:`u` if `"g_first"` and :math:`x` otherwise.
        :param torch.Tensor w: Current second variable :math:`A z` if `"g_first"` and :math:`A^\top u` otherwise.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current gStep parameters (keys `"prox_g"`, `"stepsize"` (or `"stepsize_dual"`) and `"g_param"`).
        """
        if self.g_first:
            p = x + cur_params["stepsize_dual"] * w
            return cur_prior.prox_conjugate(
                p,
                cur_params["g_param"],
                gamma=cur_params["lambda"] * cur_params["stepsize_dual"],
                lamb=cur_params["lambda"],
            )
        else:
            p = x - cur_params["stepsize"] * w
            return cur_prior.prox(
                p,
                cur_params["g_param"],
                gamma=cur_params["stepsize"] * cur_params["lambda"],
            )
