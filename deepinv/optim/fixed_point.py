import torch
import torch.nn as nn
import warnings
from tqdm import tqdm


class FixedPoint(nn.Module):
    r"""
    Fixed-point iterations module.

    This module implements the fixed-point iteration algorithm given a specific fixed-point iterator (e.g.
    proximal gradient iteration, the ADMM iteration, see :ref:`optim_iterators`), that is
    for :math:`k=1,2,...`

    .. math::
        \qquad (x_{k+1}, u_{k+1}) = \operatorname{FixedPoint}(x_k, u_k, f, g, A, y, ...) \hspace{2cm} (1)

    where :math:`f` is the data-fidelity term, :math:`g` is the prior, :math:`A` is the physics model, :math:`y` is the data.

    :Examples: This example shows how to use the :class:`FixedPoint` class to solve the problem
        :math:`\min_x 0.5*||Ax-y||_2^2 + \lambda*||x||_1` with the PGD algorithm, where A is the identity operator,
        :math:`\lambda = 1` and :math:`y = [2, 2]`.

        >>> import deepinv as dinv
        >>> # Create the measurement operator A
        >>> A = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)
        >>> A_forward = lambda v: A @ v
        >>> A_adjoint = lambda v: A.transpose(0, 1) @ v
        >>> # Define the physics model associated to this operator
        >>> physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
        >>> # Define the measurement y
        >>> y = torch.tensor([2, 2], dtype=torch.float64)
        >>> # Define the data fidelity term
        >>> data_fidelity = dinv.optim.data_fidelity.L2()
        >>> # Define the prior term
        >>> prior = dinv.optim.prior.L1Prior()
        >>> # Define the parameters of the algorithm
        >>> params_algo = {"g_param": 1.0, "stepsize": 1.0, "lambda": 1.0, "beta": 1.0}
        >>> # Choose the iterator associated to the PGD algorithm
        >>> iterator = dinv.optim.optim_iterators.PGDIteration()
        >>> # Iterate the iterator
        >>> x_init = torch.tensor([2, 2], dtype=torch.float64)  # Define initialisation of the algorithm
        >>> X = {"est": (x_init ,), "cost": []}                 # Iterates are stored in a dictionary of the form {'est': (x,z), 'cost': F}
        >>> max_iter = 50
        >>> for it in range(max_iter):
        ...     X = iterator(X, data_fidelity, prior, params_algo, y, physics)
        >>> # Return the solution
        >>> X["est"][0]
        tensor([1., 1.], dtype=torch.float64)


    :param deepinv.optim.OptimIterator iterator: function that takes as input the current iterate, as
                                        well as parameters of the optimization problem (prior, measurements, etc.)
    :param Callable update_params_fn: function that returns the parameters to be used at each iteration. Default: ``None``.
    :param Callable update_prior_fn: function that returns the prior to be used at each iteration. Default: ``None``.
    :param Callable init_iterate_fn: function that returns the initial iterate. Default: ``None``.
    :param Callable init_metrics_fn: function that returns the initial metrics. Default: ``None``.
    :param Callable check_iteration_fn: function that performs a check on the last iteration and returns a bool indicating if we can proceed to next iteration. Default: ``None``.
    :param Callable check_conv_fn: function that checks the convergence after each iteration, returns a bool indicating if convergence has been reached. Default: ``None``.
    :param int max_iter: maximum number of iterations. Default: ``50``.
    :param bool early_stop: if True, the algorithm stops when the convergence criterion is reached. Default: ``True``.
    :param bool anderson_acceleration: if True, the Anderson acceleration is used. Default: ``False``.
    :param int history_size: size of the history used for the Anderson acceleration. Default: ``5``.
    :param float beta_anderson_acc: momentum of the Anderson acceleration step. Default: ``1.0``.
    :param float eps_anderson_acc: regularization parameter of the Anderson acceleration step. Default: ``1e-4``.
    """

    def __init__(
        self,
        iterator=None,
        update_params_fn=None,
        update_data_fidelity_fn=None,
        update_prior_fn=None,
        init_iterate_fn=None,
        init_metrics_fn=None,
        update_metrics_fn=None,
        check_iteration_fn=None,
        check_conv_fn=None,
        max_iter=50,
        early_stop=True,
        anderson_acceleration=False,
        history_size=5,
        beta_anderson_acc=1.0,
        eps_anderson_acc=1e-4,
        verbose=False,
        show_progress_bar=False,
    ):
        super().__init__()
        self.iterator = iterator
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.update_params_fn = update_params_fn
        self.update_data_fidelity_fn = update_data_fidelity_fn
        self.update_prior_fn = update_prior_fn
        self.init_iterate_fn = init_iterate_fn
        self.init_metrics_fn = init_metrics_fn
        self.update_metrics_fn = update_metrics_fn
        self.check_conv_fn = check_conv_fn
        self.check_iteration_fn = check_iteration_fn
        self.anderson_acceleration = anderson_acceleration
        self.history_size = history_size
        self.beta_anderson_acc = beta_anderson_acc
        self.eps_anderson_acc = eps_anderson_acc
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar

        if self.check_conv_fn is None and self.early_stop:
            warnings.warn(
                "early_stop is set to True but no check_conv_fn has been defined."
            )
            self.early_stop = False

    def init_anderson_acceleration(self, X):
        r"""
        Initialize the Anderson acceleration algorithm.
        Code inspired from `this tutorial <http://implicit-layers-tutorial.org/deep_equilibrium_models/>`_.

        :param dict X: initial iterate.
        """
        x = X["est"][0]
        b, d, h, w = x.shape
        x_hist = torch.zeros(
            b, self.history_size, d * h * w, dtype=x.dtype, device=x.device
        )  # history of iterates.
        T_hist = torch.zeros(
            b, self.history_size, d * h * w, dtype=x.dtype, device=x.device
        )  # history of T(x_k) with T the fixed point operator.
        H = torch.zeros(
            b,
            self.history_size + 1,
            self.history_size + 1,
            dtype=x.dtype,
            device=x.device,
        )  # H in the Anderson acceleration linear system Hp = q .
        H[:, 0, 1:] = H[:, 1:, 0] = 1.0
        q = torch.zeros(
            b, self.history_size + 1, 1, dtype=x.dtype, device=x.device
        )  # q in the Anderson acceleration linear system Hp = q .
        q[:, 0] = 1
        return x_hist, T_hist, H, q

    def anderson_acceleration_step(
        self,
        it,
        X_prev,
        TX_prev,
        x_hist,
        T_hist,
        H,
        q,
        cur_data_fidelity,
        cur_prior,
        cur_params,
        *args,
    ):
        r"""
        Anderson acceleration step.

        Code inspired from `this tutorial <http://implicit-layers-tutorial.org/deep_equilibrium_models/>`_.

        :param int it: current iteration.
        :param dict X_prev: previous iterate.
        :param dict TX_prev: output of the fixed-point operator evaluated at X_prev
        :param torch.Tensor x_hist: history of last ``history-size`` iterates.
        :param torch.Tensor T_hist: history of T evlauaton at the last ``history-size``, where T is the fixed-point operator.
        :param torch.Tensor H: H in the Anderson acceleration linear system Hp = q .
        :param torch.Tensor q: q in the Anderson acceleration linear system Hp = q .
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param args: arguments for the iterator.
        """
        x_prev = X_prev["est"][0]  # current iterate Tx
        Tx_prev = TX_prev["est"][0]  # current iterate x
        b = x_prev.shape[0]  # batchsize
        x_hist[:, it % self.history_size] = x_prev.reshape(
            b, -1
        )  # prepare history of x
        T_hist[:, it % self.history_size] = Tx_prev.reshape(
            b, -1
        )  # prepare history of Tx
        m = min(it + 1, self.history_size)
        G = T_hist[:, :m] - x_hist[:, :m]
        H[:, 1 : m + 1, 1 : m + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + self.eps_anderson_acc
            * torch.eye(m, dtype=Tx_prev.dtype, device=Tx_prev.device)[None]
        )
        p = torch.linalg.solve(H[:, : m + 1, : m + 1], q[:, : m + 1])[
            :, 1 : m + 1, 0
        ]  # solve the linear system H p = q.
        x = (
            self.beta_anderson_acc * (p[:, None] @ T_hist[:, :m])[:, 0]
            + (1 - self.beta_anderson_acc) * (p[:, None] @ x_hist[:, :m])[:, 0]
        )  # Anderson acceleration step.
        x = x.view(x_prev.shape)
        F = (
            self.iterator.F_fn(x, cur_data_fidelity, cur_prior, cur_params, *args)
            if self.iterator.has_cost
            else None
        )
        est = list(TX_prev["est"])
        est[0] = x
        return {"est": est, "cost": F}

    def forward(self, *args, compute_metrics=False, x_gt=None, **kwargs):
        r"""
        Loops over the fixed-point iterator as (1) and returns the fixed point.

        The iterates are stored in a dictionary of the form ``X = {'est': (x_k, u_k), 'cost': F_k}`` where:

            * ``est`` is a tuple containing the current primal and auxiliary iterates,
            * ``cost`` is the value of the cost function at the current iterate.

        Since the prior and parameters (stepsize, regularisation parameter, etc.) can change at each iteration,
        the prior and parameters are updated before each call to the iterator.

        :param bool compute_metrics: if ``True``, the metrics are computed along the iterations. Default: ``False``.
        :param torch.Tensor x_gt: ground truth solution. Default: ``None``.
        :param args: optional arguments for the iterator. Commonly (y,physics) where ``y`` (torch.Tensor y) is the measurement and
                    ``physics`` (deepinv.physics) is the physics model.
        :param kwargs: optional keyword arguments for the iterator.
        :return tuple: ``(x,metrics)`` with ``x`` the fixed-point solution (dict) and
                    ``metrics`` the computed along the iterations if ``compute_metrics`` is ``True`` or ``None``
                    otherwise.
        """
        X = (
            self.init_iterate_fn(*args, F_fn=self.iterator.F_fn)
            if self.init_iterate_fn
            else None
        )
        metrics = (
            self.init_metrics_fn(X, x_gt=x_gt)
            if self.init_metrics_fn and compute_metrics
            else None
        )
        self.check_iteration = True
        if self.anderson_acceleration:
            self.x_hist, self.T_hist, self.H, self.q = self.init_anderson_acceleration(
                X
            )
        it = 0

        for it in tqdm(
            range(self.max_iter),
            disable=(not self.verbose or not self.show_progress_bar),
        ):
            X_prev = X
            X = self.single_iteration(
                X,
                it,
                *args,
                **kwargs,
            )

            if self.check_iteration:
                metrics = (
                    self.update_metrics_fn(metrics, X_prev, X, x_gt=x_gt)
                    if self.update_metrics_fn and compute_metrics
                    else None
                )
                if (
                    self.early_stop
                    and (self.check_conv_fn is not None)
                    and it > 1
                    and self.check_conv_fn(it, X_prev, X)
                ):
                    break
                it += 1

        return X, metrics

    def single_iteration(self, X, it, *args, **kwargs):
        cur_params = self.update_params_fn(it) if self.update_params_fn else None
        cur_data_fidelity = (
            self.update_data_fidelity_fn(it) if self.update_data_fidelity_fn else None
        )
        cur_prior = self.update_prior_fn(it) if self.update_prior_fn else None
        X_prev = X
        X = self.iterator(
            X_prev, cur_data_fidelity, cur_prior, cur_params, *args, **kwargs
        )
        if self.anderson_acceleration:
            X = self.anderson_acceleration_step(
                it,
                X_prev,
                X,
                self.x_hist,
                self.T_hist,
                self.H,
                self.q,
                cur_data_fidelity,
                cur_prior,
                cur_params,
                *args,
            )
        self.check_iteration = (
            self.check_iteration_fn(X_prev, X) if self.check_iteration_fn else True
        )
        return X if self.check_iteration else X_prev
