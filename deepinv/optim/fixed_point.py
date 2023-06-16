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
    :param function update_prior_fn: function that returns the prior to be used at each iteration. Default: None.
    :param function update_params_fn: function that returns the parameters to be used at each iteration. Default: None.
    :param function check_iteration_fn: function that performs a check on the last iteration and returns a bool indicating if we can proceed to next iteration. Default: None.
    :param function check_conv_fn: function that checks the convergence after each iteration, returns a bool indicating if convergence has been reached. Default: None.
    :param int max_iter: maximum number of iterations. Default: 50.
    :param bool early_stop: if True, the algorithm stops when the convergence criterion is reached. Default: True.
    :param str crit_conv: convergence criterion to be used for claiming convergence, either `"residual"` (residual
                          of the iterate norm) or `"cost"` (on the cost function). Default: `"residual"`
    :param float thres_conv: value of the threshold for claiming convergence. Default: `1e-05`.
    """

    def __init__(
        self,
        iterator=None,
        update_params_fn=None,
        update_prior_fn=None,
        max_iter=50,
        early_stop=True,
        init_metrics_fn=None,
        update_metrics_fn=None,
        check_iteration_fn=None,
        check_conv_fn=None,
    ):
        super().__init__()
        self.iterator = iterator
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.update_params_fn = update_params_fn
        self.update_prior_fn = update_prior_fn
        self.init_metrics_fn = init_metrics_fn
        self.update_metrics_fn = update_metrics_fn
        self.check_conv_fn = check_conv_fn
        self.check_iteration_fn = check_iteration_fn

        if self.check_conv_fn is None:
            raise Warning(
                "early_stop is set to True but no check_conv_fn has been defined. Seeting early_stop to False"
            )
            self.early_stop = False

    def forward(self, X, *args, **kwargs):
        r"""
        Loops over the fixed-point iterator as (1) and returns the fixed point.

        The iterates are stored in a dictionary of the form ``X = {'est': (x_k, u_k), 'cost': F_k}`` where:
            - `est` is a tuple containing the current primal and dual iterates,
            - `cost` is the value of the cost function at the current iterate.

        Since the prior and parameters (stepsize, regularisation parameter, etc.) can change at each iteration,
        the prior and parameters are updated before each call to the iterator.

        :param dict X: dictionary containing the current iterate.
        :param args: optional arguments for the iterator.
        :param kwargs: optional keyword arguments for the iterator.
        :return: the fixed-point.
        """
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


class AndersonAcceleration(FixedPoint):
    r"""
    Anderson Acceleration algorithm for fixed-point algorithms.

    Considering a fixed-point algorithm of the form $x_{k+1} = T(x_k)$, the Anderson algorithm (see
    `<https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf>`_.) is defined as:

    .. math::

        x_{k+1} = (1-\beta) \sum_{i=0}^{m} \alpha_i x_{k-m+i} + \beta \sum_{i=0}^{m} \alpha_i T(x_{k-m+i})


    where :math:`T` is the fixed-point iterator and the coefficients :math:`\alpha_i` are such that :math:`\sum_{i=0}^{m} \alpha_i = 1`.

    :param int history_size: number of previous iterates to be used for the acceleration (parameter :math:`m` in the above equation). Default: 5.
    :param float ridge: ridge parameter for the least-squares problem. Default: 1e-4.
    :param float or list beta: parameter :math:`\beta` in the above equation. If a float is provided, the same value is used for all iterations. If a list is provided, the value is updated at each iteration. Default: 1.0.
    :param kwargs: additional keyword arguments for FixedPoint.
    """

    def __init__(self, history_size=5, ridge=1e-4, beta=1.0, **kwargs):
        super(AndersonAcceleration, self).__init__(**kwargs)
        self.history_size = history_size
        if isinstance(beta, float):
            beta = [beta] * self.max_iter
        self.beta = beta
        self.ridge = ridge

    def forward(self, x, init_params, *args):
        r"""
        Computes the fixed point of :math:`x_{k+1}=T(x_k)` using Anderson acceleration.

        :param dict x: dictionary containing the current iterate.
        :param dict init_params: dictionary containing the initial parameters.
        :param args: optional arguments for the iterator.
        :return: a tuple containing the fixed-point.
        """
        cur_params = init_params
        init = x["est"][0]
        B, C, H, W = init.shape
        X = torch.zeros(
            B, self.history_size, C * H * W, dtype=init.dtype, device=init.device
        )
        F = torch.zeros(
            B, self.history_size, C * H * W, dtype=init.dtype, device=init.device
        )
        X[:, 0] = init.reshape(B, -1)
        F[:, 0] = self.iterator(init, 0, *args)[0].reshape(B, -1)
        X[:, 1] = F[:, 0]
        x = self.iterator(F[:, 0].reshape(init.shape), 1, *args)[0]
        F[:, 1] = x.reshape(B, -1)

        H = torch.zeros(
            B,
            self.history_size + 1,
            self.history_size + 1,
            dtype=init.dtype,
            device=init.device,
        )
        H[:, 0, 1:] = H[:, 1:, 0] = 1
        y = torch.zeros(B, self.history_size + 1, dtype=init.dtype, device=init.device)
        y[:, 0] = 1
        for it in range(2, self.max_iter):
            n = min(it, self.history_size)
            G = F[:, :n] - X[:, :n]
            H[:, 1 : n + 1, 1 : n + 1] = torch.bmm(
                G, G.transpose(1, 2)
            ) + self.ridge * torch.eye(
                n, dtype=init.dtype, device=init.device
            ).unsqueeze(
                0
            )
            alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
                :, 1 : n + 1
            ]
            X[:, it % self.history_size] = (
                self.beta[it] * (alpha[:, None] @ F[:, :n])[:, 0]
                + (1 - self.beta[it]) * (alpha[:, None] @ X[:, :n])[:, 0]
            )
            F[:, it % self.history_size] = self.iterator(
                X[:, it % self.history_size].reshape(init.shape), it, *args
            )[0].reshape(B, -1)
            x_prev = X[:, it % self.history_size].reshape(init.shape)
            x = F[:, it % self.history_size].reshape(init.shape)
            if (
                self.check_conv_fn(
                    x_prev, x, it, self.crit_conv, self.thres_conv, verbose=self.verbose
                )
                and it > 1
            ):
                self.has_converged = True
                if self.early_stop:
                    if self.verbose:
                        print("Convergence reached at iteration ", it)
                    break
            if it < self.max_iter - 1:
                cur_params = self.update_params(cur_params, it + 1, x, x_prev)
        return (x,)
