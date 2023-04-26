import torch
import torch.nn as nn
from deepinv.optim.utils import check_conv


class FixedPoint(nn.Module):
    """
    Fixed-point iterations module.

    This module implements the fixed-point iteration algorithm given a specific fixed-point iterator (e.g.
    proximal gradient iteration, the ADMM iteration, see :meth:`deepinv.optim.optim_iterators`), that is
    for :math:`k=1,2,...`

    .. math::
        \qquad (x_{k+1}, u_{k+1}) = \operatorname{FixedPoint}(x_k, u_k, f, g, A, y, ...) \hspace{2cm} (1)



    ::

            # Generate the data
            x = torch.ones(1, 1, 1, 3)
            A = torch.Tensor([[2, 0, 0], [0, -0.5, 0], [0, 0, 1]])
            A_forward = lambda v: A @ v
            A_adjoint = lambda v: A.transpose(0, 1) @ v

            # Define the physics model associated to this operator and the data
            physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
            y = physics.A(x)

            # Select the data fidelity term
            data_fidelity = L2()

            # Specify the prior and the algorithm parameters
            model_spec = {"name": "waveletprior", "args": {"wv": "db8", "level": 3, "device": device}}
            prior = {"prox_g": Denoiser(model_spec)}
            params_algo = {"stepsize": 0.1, "g_param": 1.0}

            # Choose the iterator associated to a specific algorithm
            iterator = PGDIteration(data_fidelity=data_fidelity)

            # Create the optimizer
            optimizer = BaseOptim(
                iterator,
                params_algo=params_algo,
                prior=prior,
                max_iter=max_iter,
            )

            # Run the optimization algorithm
            x = optimizer(y, physics)


    :param deepinv.optim.optim_iterators.optim_iterator iterator: function that takes as input the current iterate, as
                                        well as parameters of the optimisation problem (prior, measurements, etc.)
    :param function update_prior_fn: function that returns the prior to be used at each iteration. Default: None.
    :param function update_params_fn_pre: function that returns the parameters to be used at each iteration. Default: None.
    :param int max_iter: maximum number of iterations. Default: 50.
    :param bool early_stop: if True, the algorithm stops when the convergence criterion is reached. Default: True.
    :param str crit_conv: convergence criterion to be used for claiming convergence, either `"residual"` (residual
                          of the iterate norm) or `"cost"` (on the cost function). Default: `"residual"`
    :param float thres_conv: value of the threshold for claiming convergence. Default: `1e-05`.
    :param bool verbose: if True, prints the current iteration number and the current value of the
                            stopping criterion. Default: False.
    """

    def __init__(
        self,
        iterator=None,
        update_params_fn_pre=None,
        update_prior_fn=None,
        max_iter=50,
        early_stop=True,
        init_metrics_fn=None,
        update_metrics_fn=None,
        check_conv_fn=None,
    ):
        super().__init__()
        self.iterator = iterator
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.update_params_fn_pre = update_params_fn_pre
        self.update_prior_fn = update_prior_fn
        self.init_metrics_fn = init_metrics_fn
        self.update_metrics_fn = update_metrics_fn
        self.check_conv_fn = check_conv_fn

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
        X_prev = None
        metrics = self.init_metrics_fn(X, **kwargs)
        for it in range(self.max_iter):
            cur_prior = self.update_prior_fn(it)
            cur_params = self.update_params_fn_pre(it, X, X_prev)
            X_prev = X
            X = self.iterator(X, cur_prior, cur_params, *args)
            metrics = self.update_metrics_fn(metrics, X_prev, X, **kwargs)
            if self.early_stop and self.check_conv_fn(it, X_prev, X) and it > 1:
                break
        return X, metrics


class AndersonAcceleration(FixedPoint):
    """
    Anderson Acceleration for accelerated fixed-point resolution.

    The implementation is strongly inspired from http://implicit-layers-tutorial.org/deep_equilibrium_models/.
    Foward is called with init a tuple (x,) with x the initialization tensor of shape BxCxHxW and iterator optional arguments.

    :param int history_size: size of the history used for the acceleration. Default: 5.
    :param float ridge: ridge regularization in solver. Default: 1e-4.
    :param float beta: momentum in Anderson updates. Default: 1.0.
    :param kwargs: optional keyword arguments for the iterator.
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
        Computes the fixed-point iterations with Anderson acceleration.

        :param dict x: dictionary with key "est" and value the initial estimate.
        :param init_params: initial parameters for the iterator.
        :param args: optional arguments for the iterator.
        :return: the fixed-point iterate.
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
                check_conv(
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
