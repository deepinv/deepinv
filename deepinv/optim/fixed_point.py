import torch
import torch.nn as nn


class FixedPoint(nn.Module):
    """
    Fixed-point iterations module.

    This module implements the fixed-point iteration algorithm given a specific fixed-point iterator (e.g.
    proximal gradient iteration, the ADMM iteration, see :meth:`deepinv.optim.optim_iterators`), that is
    for :math:`k=1,2,...`

    .. math::
        \qquad (x_{k+1}, u_{k+1}) = \operatorname{FixedPoint}(x_k, u_k, f, g, A, y, ...) \hspace{2cm} (1)


    where :math:`f` is the data-fidelity term, :math:`g` is the prior, :math:`A` is the physics model, :math:`y` is the
    data.


    ::

        # This example shows how to use the FixedPoint class to solve the problem
        #                min_x 0.5*lambda*||Ax-y||_2^2 + ||x||_1
        # with the PGD algorithm, where A is the identity operator, lambda = 1 and y = [2, 2].

        # Create the measurement operator A
        A = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)
        A_forward = lambda v: A @ v
        A_adjoint = lambda v: A.transpose(0, 1) @ v

        # Define the physics model associated to this operator
        physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

        # Define the measurement y
        y = torch.tensor([2, 2], dtype=torch.float64)

        # Define the data fidelity term
        data_fidelity = L2()

        # Define the proximity operator of the prior and store it in a dictionary
        def prox_g(x, g_param=0.1):
            return torch.sign(x) * torch.maximum(x.abs() - g_param, torch.tensor([0]))

        prior = {"prox_g": prox_g}

        # Define the parameters of the algorithm
        params = {"g_param": 1.0, "stepsize": 1.0, "lambda": 1.0}

        # Choose the iterator associated to the PGD algorithm
        iterator = PGDIteration(data_fidelity=data_fidelity)

        # Iterate the iterator
        x_init = torch.tensor([2, 2], dtype=torch.float64)  # Define initialisation of the algorithm
        X = {"est": (x_init ,), "cost": []}                 # Iterates are stored in a dictionary of the form {'est': (x,z), 'cost': F}

        max_iter = 50
        for it in range(max_iter):
            X = iterator(X,  prior, params, y, physics)

        # Return the solution
        sol = X["est"][0]  # sol = [1, 1]


    :param deepinv.optim.optim_iterators.optim_iterator iterator: function that takes as input the current iterate, as
                                        well as parameters of the optimisation problem (prior, measurements, etc.)
    :param function update_params_fn: function that returns the parameters to be used at each iteration. Default: None.
    :param function update_prior_fn: function that returns the prior to be used at each iteration. Default: None.
    :param function init_iterate_fn: function that returns the initial iterate. Default: None.
    :param function init_metrics_fn: function that returns the initial metrics. Default: None.
    :param function check_iteration_fn: function that performs a check on the last iteration and returns a bool indicating if we can proceed to next iteration. Default: None.
    :param function check_conv_fn: function that checks the convergence after each iteration, returns a bool indicating if convergence has been reached. Default: None.
    :param int max_iter: maximum number of iterations. Default: 50.
    :param bool early_stop: if True, the algorithm stops when the convergence criterion is reached. Default: True.
    """

    def __init__(
        self,
        iterator=None,
        update_params_fn=None,
        update_prior_fn=None,
        init_iterate_fn=None,
        init_metrics_fn=None,
        update_metrics_fn=None,
        check_iteration_fn=None,
        check_conv_fn=None,
        max_iter=50,
        early_stop=True,
    ):
        super().__init__()
        self.iterator = iterator
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.update_params_fn = update_params_fn
        self.update_prior_fn = update_prior_fn
        self.init_iterate_fn = init_iterate_fn
        self.init_metrics_fn = init_metrics_fn
        self.update_metrics_fn = update_metrics_fn
        self.check_conv_fn = check_conv_fn
        self.check_iteration_fn = check_iteration_fn

        if self.check_conv_fn is None and self.early_stop:
            raise Warning(
                "early_stop is set to True but no check_conv_fn has been defined. Seeting early_stop to False"
            )
            self.early_stop = False

    def forward(self, *args, **kwargs):
        r"""
        Loops over the fixed-point iterator as (1) and returns the fixed point.

        The iterates are stored in a dictionary of the form ``X = {'est': (x_k, u_k), 'cost': F_k}`` where:
            - `est` is a tuple containing the current primal and auxiliary iterates,
            - `cost` is the value of the cost function at the current iterate.

        Since the prior and parameters (stepsize, regularisation parameter, etc.) can change at each iteration,
        the prior and parameters are updated before each call to the iterator.

        :param args: optional arguments for the iterator.
        :param kwargs: optional keyword arguments for the iterator.
        :return: the fixed-point.
        """
        X = (
            self.init_iterate_fn(*args, F_fn=self.iterator.F_fn)
            if self.init_iterate_fn
            else None
        )
        metrics = self.init_metrics_fn(X, **kwargs) if self.init_metrics_fn else None
        it = 0
        while it < self.max_iter:
            cur_params = self.update_params_fn(it) if self.update_params_fn else None
            cur_prior = self.update_prior_fn(it) if self.update_prior_fn else None
            X_prev = X
            X = self.iterator(X_prev, cur_prior, cur_params, *args)
            check_iteration = (
                self.check_iteration_fn(X_prev, X) if self.check_iteration_fn else True
            )
            if check_iteration:
                metrics = (
                    self.update_metrics_fn(metrics, X_prev, X, **kwargs)
                    if self.update_metrics_fn
                    else None
                )
                if (
                    (self.early_stop and it > 1) or (it == self.max_iter - 1)
                ) and self.check_conv_fn(it, X_prev, X):
                    break
                it += 1
            else:
                X = X_prev
        return X, metrics
