from .optim_iterator import OptimIterator, fStep, gStep


class GDIteration(OptimIterator):
    r"""
    Iterator for Gradient Descent.

    Class for a single iteration of the gradient descent (GD) algorithm for minimising :math:`\lambda f(x) + g(x)`.

    The iteration is given by


    .. math::
        \begin{equation*}
        \begin{aligned}
        v_{k} &= \nabla f(x_k) + \nabla g(x_k) \\
        x_{k+1} &= x_k-\gamma v_{k}
        \end{aligned}
        \end{equation*}


   where :math:`\gamma` is a stepsize.
    """

    def __init__(self, **kwargs):
        super(GDIteration, self).__init__(**kwargs)
        self.g_step = gStepGD(**kwargs)
        self.f_step = fStepGD(**kwargs)
        self.requires_grad_g = True

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        Single gradient descent iteration on the objective :math:`f(x) + \lambda g(x)`.

        :param dict X: Dictionary containing the current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
         :return: Dictionary `{'iterate' : x,  'estimate': z , 'cost': F}` containing the updated iterate, estimate and cost value.
        """
        x_prev = X["iterate"]
        grad = cur_params["stepsize"] * (
            self.g_step(x_prev, cur_prior, cur_params)
            + self.f_step(x_prev, cur_data_fidelity, cur_params, y, physics)
        )
        x = x_prev - grad
        iterate = x
        estimate = self.get_estimate_from_iterate(
            iterate, cur_data_fidelity, cur_prior, cur_params, y, physics
        )
        cost = (
            self.cost_fn(estimate, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"iterate": iterate, "estimate": estimate, "cost": cost}


class fStepGD(fStep):
    r"""
    GD fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepGD, self).__init__(**kwargs)

    def forward(self, x, cur_data_fidelity, cur_params, y, physics):
        r"""
        Single gradient descent iteration on the data fit term :math:`f`.

        :param torch.Tensor x: current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        """
        return cur_data_fidelity.grad(x, y, physics)


class gStepGD(gStep):
    r"""
    GD gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepGD, self).__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`\lambda g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        return cur_params["lambda"] * cur_prior.grad(x, cur_params["g_param"])
