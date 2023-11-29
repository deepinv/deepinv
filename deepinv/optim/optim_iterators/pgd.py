from .optim_iterator import OptimIterator, fStep, gStep
from .utils import gradient_descent_step


class PGDIteration(OptimIterator):
    r"""
    Iterator for proximal gradient descent.

    Class for a single iteration of the Proximal Gradient Descent (PGD) algorithm for minimising :math:`\lambda f(x) + g(x)`.

    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= x_k - \lambda \gamma \nabla f(x_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma g}(u_k),
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` is a stepsize that should satisfy :math:`\lambda \gamma \leq 2/\operatorname{Lip}(\|\nabla f\|)`.

    """

    def __init__(self, **kwargs):
        super(PGDIteration, self).__init__(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)
        if self.g_first:
            self.requires_grad_g = True
        else:
            self.requires_prox_g = True


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
            # if cur_params["lambda"] >= 2:
            #     raise ValueError("lambda must be smaller than 2")
            grad = (
                cur_params["lambda"]
                * cur_params["stepsize"]
                * cur_data_fidelity.grad(x, y, physics)
            )
            return gradient_descent_step(x, grad)
        else:
            return cur_data_fidelity.prox(
                x, y, physics, gamma=cur_params["lambda"] * cur_params["stepsize"]
            )


class gStepPGD(gStep):
    r"""
    PGD gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepPGD, self).__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        if not self.g_first:
            return cur_prior.prox(
                x, cur_params["g_param"], gamma=cur_params["stepsize"]
            )
        else:
            grad = cur_params["stepsize"] * cur_prior.grad(x, cur_params["g_param"])
            return gradient_descent_step(x, grad)
