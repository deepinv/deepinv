import torch

from .optim_iterator import OptimIterator, fStep, gStep

class CPIteration(OptimIterator):
    r"""
    Single iteration of the Chambolle-Pock algorithm.

    Class for a single iteration of the `Chambolle-Pock <https://hal.science/hal-00490826/document>`_ Primal-Dual (PD)
    algorithm for minimising :math:`\lambda \datafid{Ax}{y} + g(x)`. Our implementation corresponds to
    Algorithm 1 of `<https://hal.science/hal-00609728v4/document>`_.

    If the attribute `"g_first"` is set to False (by default), the iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\sigma (\lambda f)^*}(u_k + \sigma K z_k) \\
        x_{k+1} &= \operatorname{prox}_{\tau g}(x_k-\tau K^\top u_{k+1}) \\
        z_{k+1} &= x_{k+1} + \beta(x_{k+1}-x_k) \\
        \end{aligned}
        \end{equation*}

    where :math:`(\lambda f)^*` is the Fenchel-Legendre conjugate of :math:`\lambda f`, :math:`\beta>0` is a relaxation parameter, and :math:`\sigma` and :math:`\tau` are step-sizes that should
    satisfy :math:`\sigma \tau \|K\|^2 \leq 1`.

    If the attribute `"g_first"` is set to True, the functions :math:`f` and :math:`g` are inverted in the previous iteration.
    """

    def __init__(self, K=None, **kwargs):
        super(CPIteration, self).__init__(**kwargs)
        self.g_step = gStepCP(**kwargs)
        self.f_step = fStepCP(**kwargs)
        self.K = K if K is not None else CustomLinearOperator()

    def forward(self, X, cur_prior, cur_params, y, physics):
        r"""
        Single iteration of the Chambolle-Pock algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: dictionary containing the current parameters of the model.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev, z_prev, u_prev = X["est"]
        if self.g_first:
            u = self.g_step(u_prev, self.K(z_prev), cur_prior, cur_params)
            x = self.f_step(x_prev, self.K.adjoint(u), y, physics, cur_params)
        else:
            u = self.f_step(u_prev, self.K(z_prev), y, physics, cur_params)
            x = self.g_step(x_prev, self.K.adjoint(u), cur_prior, cur_params)
        z = x + self.beta * (x - x_prev)
        F = self.F_fn(x, cur_prior, cur_params, y, physics) if self.F_fn else None

        return {"est": (x, u), "cost": F}


class fStepCP(fStep):
    r"""
    Chambolle-Pock fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepCP, self).__init__(**kwargs)

    def forward(self, x, w, y, physics, cur_params):
        r"""
        Single Chambolle-Pock iteration step on the data-fidelity term :math:`\lambda f`.

        :param torch.Tensor x: Current first variable :math:`x` if `"g_first"` and :math:`u` otherwise.
        :param torch.Tensor w: Current second variable :math:`A^\top u` if `"g_first"` and :math:`A z` otherwise.
        :param torch.Tensor y: Input data.
        :param dict cur_params: Dictionary containing the current fStep parameters (keys `"stepsize"` and `"lambda"`).
        """
        if self.g_first:
            p = x - cur_params["stepsize"] * w
            return self.data_fidelity.prox(p, y, cur_params["lambda"] * cur_params["stepsize"])
        else:
            p = x + cur_params["sigma"] * w
            return self.data_fidelity.prox_conjugate(p, y, cur_params["sigma"], lamb = cur_params["lambda"])

class gStepCP(gStep):
    r"""
    Chambolle-Pock gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepCP, self).__init__(**kwargs)

    def forward(self, x, w, cur_prior, cur_params):
        r"""
        Single Chambolle-Pock iteration step on the prior term :math:`g`.

        :param torch.Tensor x: Current first variable :math:`u` if `"g_first"` and :math:`x` otherwise.
        :param torch.Tensor w: Current second variable :math:`A z` if `"g_first"` and :math:`A^\top u` otherwise.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current gStep parameters (keys `"prox_g"`, `"stepsize"` and `"g_param"`).
        """
        if self.g_first:
            p = x + cur_params["sigma"] * w
            return cur_prior.prox_conjugate(p, cur_params["sigma"], cur_params["g_param"])
        else: 
            p = x - cur_params["stepsize"] * w
            return cur_prior.prox(p, cur_params["stepsize"], cur_params["g_param"])


class CustomLinearOperator(torch.nn.Module):
    r"""
    A base class for simple user-defined linear operators.

    The user needs to provide the forward operator :math:`L` and its adjoint (backward) :math:`L^{\top}`.

    :param callable fwd: forward operator function which maps an image to the observed measurements :math:`x\mapsto y`.
    :param callable bwd: adjoint of the forward operator, which should verify the adjointness test.

    """

    def __init__(
        self,
        fwd_op=lambda x: x,
        bwd_op=lambda x: x
    ):
        super().__init__()

        self.fwd_op = fwd_op
        self.bwd_op = bwd_op

    def forward(self, x):
        r"""
        Computes the forward operator :math:`L(x)`.

        :param torch.tensor x: input.
        :return: (torch.tensor) :math:`L(x)`.

        """
        return self.fwd_op(x)


    def adjoint(self, y):
        r"""
        Computes the adjoint of the linear operator :math:`L`, i.e. :math:`L^{\top}(y)`.

        :param torch.tensor y: input.
        :return: (torch.tensor) :math:`\tilde{x} = L^{\top}y`.

        """
        return self.bwd_op(y)
