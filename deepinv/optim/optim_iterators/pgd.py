from .optim_iterator import OptimIterator, fStep, gStep
from .utils import gradient_descent_step


class PGDIteration(OptimIterator):
    r"""
    Iterator for proximal gradient descent.

    Class for a single iteration of the Proximal Gradient Descent (PGD) algorithm for minimizing :math:` f(x) + \lambda g(x)`.

    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= x_k -  \gamma \nabla f(x_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma \lambda g}(u_k),
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` is a stepsize that should satisfy :math:` \gamma \leq 2/\operatorname{Lip}(\|\nabla f\|)`.

    """

    def __init__(self, **kwargs):
        super(PGDIteration, self).__init__(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)
        if self.g_first:
            self.requires_grad_g = True
        else:
            self.requires_prox_g = True


class FISTAIteration(OptimIterator):
    r"""
    Iterator for fast iterative soft-thresholding (FISTA).

    Class for a single iteration of the FISTA algorithm for minimizing :math:` f(x) + \lambda g(x)` as proposed by
    `Chambolle \& Dossal <https://inria.hal.science/hal-01060130v3/document>`_.

    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= x_k -  \gamma \nabla f(z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma \lambda g}(u_k) \\
        z_{k+1} &= x_{k+1} + \alpha_k (x_{k+1} - x_k),
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` is a stepsize that should satisfy :math:` \gamma \leq 1/\operatorname{Lip}(\|\nabla f\|)` and
    :math:`\alpha_k = (t_k + a - 1)/(t_k + a)`.
    """

    def __init__(self, a=3, **kwargs):
        super(FISTAIteration, self).__init__(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)
        self.a = a
        if self.g_first:
            self.requires_grad_g = True
        else:
            self.requires_prox_g = True

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        Forward pass of an iterate of the FISTA algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev, z_prev = X["est"][0], X["est"][1]
        k = 2 if "it" not in X else X["it"]
        alpha = (k - 1) / (k + self.a)

        if not self.g_first:
            z = self.f_step(z_prev, cur_data_fidelity, cur_params, y, physics)
            x = self.g_step(z, cur_prior, cur_params)
        else:
            z = self.g_step(z_prev, cur_prior, cur_params)
            x = self.f_step(z, cur_data_fidelity, cur_params, y, physics)

        z = x + alpha * (x - x_prev)

        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )

        return {"est": (x, z), "cost": F, "it": k + 1}


class fStepPGD(fStep):
    r"""
    PGD fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepPGD, self).__init__(**kwargs)

    def forward(self, x, cur_data_fidelity, cur_params, y, physics):
        r"""
         Single PGD iteration step on the data-fidelity term :math:`f`.

         :param torch.Tensor x: Current iterate :math:`x_k`.
         :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
         :param torch.Tensor y: Input data.
         :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        """
        if not self.g_first:
            grad = cur_params["stepsize"] * cur_data_fidelity.grad(x, y, physics)
            return gradient_descent_step(x, grad)
        else:
            return cur_data_fidelity.prox(x, y, physics, gamma=cur_params["stepsize"])


class gStepPGD(gStep):
    r"""
    PGD gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepPGD, self).__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`\lambda g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        if not self.g_first:
            return cur_prior.prox(
                x,
                cur_params["g_param"],
                gamma=cur_params["lambda"] * cur_params["stepsize"],
            )
        else:
            grad = (
                cur_params["lambda"]
                * cur_params["stepsize"]
                * cur_prior.grad(x, cur_params["g_param"])
            )
            return gradient_descent_step(x, grad)
