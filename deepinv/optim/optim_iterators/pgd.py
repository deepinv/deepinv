from .optim_iterator import OptimIterator, fStep, gStep
from deepinv.optim.bregman import BregmanL2


class PGDIteration(OptimIterator):
    r"""
    Iterator for proximal gradient descent.

    Class for a single iteration of the Proximal Gradient Descent (PGD) algorithm for minimizing :math:`f(x) + \lambda \regname(x)`.

    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= x_k -  \gamma \nabla f(x_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(u_k),
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` is a stepsize that should satisfy :math:`\gamma \leq 2/\operatorname{Lip}(\|\nabla f\|)`.

    """

    def __init__(self, **kwargs):
        super(PGDIteration, self).__init__(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)


class FISTAIteration(OptimIterator):
    r"""
    Iterator for fast iterative soft-thresholding (FISTA).

    Class for a single iteration of the FISTA algorithm for minimizing :math:`f(x) + \lambda \regname(x)` as proposed by
    `Chambolle \& Dossal <https://inria.hal.science/hal-01060130v3/document>`_.

    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= z_k -  \gamma \nabla f(z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma \lambda \regname}(u_k) \\
        z_{k+1} &= x_{k+1} + \alpha_k (x_{k+1} - x_k),
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` is a stepsize that should satisfy :math:`\gamma \leq 1/\operatorname{Lip}(\|\nabla f\|)` and
    :math:`\alpha_k = (k+a-1)/(k+a)`, with :math:`a` a parameter that should be strictly greater than 2.
    """

    def __init__(self, a=3, **kwargs):
        super(FISTAIteration, self).__init__(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)
        self.a = a
        if self.a != 3:
            raise DeprecationWarning(
                "setting the a parameter in the FISTAIteration is deprecated, it should now be set in the cur_params dictionary instead."
            )

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        Forward pass of an iterate of the FISTA algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev, z_prev = X["est"][0], X["est"][1]
        k = 0 if "it" not in X else X["it"]
        a = cur_params["a"]
        alpha = (k + a - 1) / (k + a)

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
            and self.F_fn is not None
            and cur_data_fidelity is not None
            and cur_prior is not None
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
         :param deepinv.physics.Physics physics: Instance of the physics modeling the data-fidelity term.
        """
        if not self.g_first:
            grad = cur_params["stepsize"] * cur_data_fidelity.grad(x, y, physics)
            return x - grad
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
        Single iteration step on the prior term :math:`\lambda \regname`.

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
            return x - grad


class PMDIteration(OptimIterator):
    r"""
    Iterator for Proximal Mirror Descent (PMD).

    Class for a single iteration of the Proximal Mirror Descent (PMD) algorithm for minimizing :math:`f(x) + \lambda \regname(x)`.

    For a given Bregman convex potential :math:`h`, the iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= \nabla h^*(\nabla h(x_k) - \gamma \nabla f(x_k)) \\
        x_{k+1} &= \operatorname{prox^h}_{\gamma \lambda \regname}(u_k)
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` is a stepsize that should satisfy :math:`\gamma \leq 2/L` with :math:`L` verifying :math:`Lh-f` is convex.
    The potential :math:`h` should be specified in the cur_params dictionary.
    """

    def __init__(self, bregman_potential=BregmanL2(), **kwargs):
        super(PMDIteration, self).__init__(**kwargs)
        self.bregman_potential = bregman_potential
        self.g_step = gStepPMD(bregman_potential=bregman_potential, **kwargs)
        self.f_step = fStepPMD(bregman_potential=bregman_potential, **kwargs)


class fStepPMD(fStep):
    r"""
    Proximal Mirror Descent fStep module.
    """

    def __init__(self, bregman_potential=BregmanL2(), **kwargs):
        super(fStepPMD, self).__init__(**kwargs)
        self.bregman_potential = bregman_potential

    def forward(self, x, cur_data_fidelity, cur_params, y, physics):
        r"""
         Single Proximal Mirror Descent iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the data-fidelity term.
        :param deepinv.optim.Bregman Bregman potential used for Bregman optimization algorithms such as Mirror Descent.
        """
        if not self.g_first:
            grad = cur_params["stepsize"] * cur_data_fidelity.grad(x, y, physics)
            return self.bregman_potential.grad_conj(
                self.bregman_potential.grad(x) - grad
            )
        else:
            return cur_data_fidelity.bregman_prox(
                x, self.bregman_potential, y, physics, gamma=cur_params["stepsize"]
            )


class gStepPMD(gStep):
    r"""
    Proximal Mirror Descent gStep module.
    """

    def __init__(self, bregman_potential=BregmanL2(), **kwargs):
        super(gStepPMD, self).__init__(**kwargs)
        self.bregman_potential = bregman_potential

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`\lambda g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param deepinv.optim.Bregman Bregman potential used for Bregman optimization algorithms such as Mirror Descent.
        """
        if not self.g_first:
            return cur_prior.bregman_prox(
                x,
                self.bregman_potential,
                cur_params["g_param"],
                gamma=cur_params["lambda"] * cur_params["stepsize"],
            )
        else:
            grad = (
                cur_params["lambda"]
                * cur_params["stepsize"]
                * cur_prior.grad(x, cur_params["g_param"])
            )
            return self.bregman_potential.grad_conj(
                self.bregman_potential.grad(x) - grad
            )
