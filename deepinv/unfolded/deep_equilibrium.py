import torch
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.optim_iterators import *
from deepinv.unfolded.unfolded import BaseUnfold
from deepinv.optim.optimizers import create_iterator
from deepinv.optim.data_fidelity import L2


class BaseDEQ(BaseUnfold):
    r"""
    Base class for deep equilibrium (DEQ) algorithms. Child of :class:`deepinv.unfolded.BaseUnfold`.

    Enables to turn any fixed-point algorithm into a DEQ algorithm, i.e. an algorithm
    that can be virtually unrolled infinitely leveraging the implicit function theorem.
    The backward pass is performed using fixed point iterations to find solutions of the fixed-point equation

    .. math::

        \begin{equation}
        v = \left(\frac{\partial \operatorname{FixedPoint}(x^\star)}{\partial x^\star} \right )^{\top} v + u.
        \end{equation}

    where :math:`u` is the incoming gradient from the backward pass,
    and :math:`x^\star` is the equilibrium point of the forward pass.

    See `this tutorial <http://implicit-layers-tutorial.org/deep_equilibrium_models/>`_ for more details.

    .. note::

        For now DEQ is only possible with PGD, HQS and GD optimization algorithms.

    :param bool jacobian_free: Does not inverse the Jacobian but simply uses ``v=u``.
    :param int max_iter_backward: Maximum number of backward iterations. Default: ``50``.
    :param bool anderson_acceleration_backward: if True, the Anderson acceleration is used at iteration of fixed-point algorithm for computing the backward pass. Default: ``False``.
    :param int history_size_backward: size of the history used for the Anderson acceleration for the backward pass. Default: ``5``.
    :param float beta_anderson_acc_backward: momentum of the Anderson acceleration step for the backward pass. Default: ``1.0``.
    :param float eps_anderson_acc_backward: regularization parameter of the Anderson acceleration step for the backward pass. Default: ``1e-4``.
    """

    def __init__(
        self,
        *args,
        jacobian_free=False,
        max_iter_backward=50,
        anderson_acceleration_backward=False,
        history_size_backward=5,
        beta_anderson_acc_backward=1.0,
        eps_anderson_acc_backward=1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.jacobian_free = jacobian_free
        self.max_iter_backward = max_iter_backward
        self.anderson_acceleration = anderson_acceleration_backward
        self.history_size = history_size_backward
        self.beta_anderson_acc = beta_anderson_acc_backward
        self.eps_anderson_acc = eps_anderson_acc_backward

    def forward(self, y, physics, x_gt=None, compute_metrics=False, **kwargs):
        r"""
        The forward pass of the DEQ algorithm. Compared to :class:`deepinv.unfolded.BaseUnfold`, the backward algorithm is performed using fixed point iterations.

        :param torch.Tensor y: Input tensor.
        :param deepinv.physics.Physics physics: Physics object.
        :param torch.Tensor x_gt: (optional) ground truth image, for plotting the PSNR across optim iterations.
        :param bool compute_metrics: whether to compute the metrics or not. Default: ``False``.
        :return: If ``compute_metrics`` is ``False``,  returns (:class:`torch.Tensor`) the output of the algorithm.
                Else, returns (:class:`torch.Tensor`, dict) the output of the algorithm and the metrics.
        """
        with torch.no_grad():  # Perform the forward pass without gradient tracking
            X, metrics = self.fixed_point(
                y, physics, x_gt=x_gt, compute_metrics=compute_metrics, **kwargs
            )

        # Once, at the equilibrium point, performs one additional iteration with gradient tracking.
        cur_data_fidelity = (
            self.update_data_fidelity_fn(self.max_iter - 1)
            if self.update_data_fidelity_fn
            else None
        )
        cur_prior = (
            self.update_prior_fn(self.max_iter - 1) if self.update_prior_fn else None
        )
        cur_params = (
            self.update_params_fn(self.max_iter - 1) if self.update_params_fn else None
        )
        x = self.fixed_point.iterator(
            X, cur_data_fidelity, cur_prior, cur_params, y, physics, **kwargs
        )["est"][0]

        # Then we perform automatic differentiation.
        # First another iteration for jacobian computation via automatic differentiation.
        x0 = x.clone().detach().requires_grad_()
        f0 = self.fixed_point.iterator(
            {"est": (x0,)},
            cur_data_fidelity,
            cur_prior,
            cur_params,
            y,
            physics,
            **kwargs,
        )["est"][0]

        # Add a backwards hook that takes the incoming backward gradient `X["est"][0]` and solves the fixed point equation
        def backward_hook(grad):
            class backward_iterator(OptimIterator):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

                def forward(self, X, *args, **kwargs):
                    return {
                        "est": (
                            torch.autograd.grad(f0, x0, X["est"][0], retain_graph=True)[
                                0
                            ]
                            + grad,
                        )
                    }

            # Use the :class:`deepinv.optim.fixed_point.FixedPoint` class to solve the fixed point equation
            def init_iterate_fn(y, physics, F_fn=None):
                return {"est": (grad,)}  # initialize the fixed point algorithm.

            if not self.jacobian_free:
                backward_FP = FixedPoint(
                    backward_iterator(),
                    init_iterate_fn=init_iterate_fn,
                    max_iter=self.max_iter_backward,
                    check_conv_fn=self.check_conv_fn,
                    anderson_acceleration=self.anderson_acceleration,
                    history_size=self.history_size,
                    beta_anderson_acc=self.beta_anderson_acc,
                    eps_anderson_acc=self.eps_anderson_acc,
                )
                return backward_FP({"est": (grad,)}, None)[0]["est"][0]
            else:
                return grad

        if x.requires_grad:
            x.register_hook(backward_hook)

        if compute_metrics:
            return x, metrics
        else:
            return x


def DEQ_builder(
    iteration,
    params_algo={"lambda": 1.0, "stepsize": 1.0},
    data_fidelity=L2(),
    prior=None,
    F_fn=None,
    g_first=False,
    bregman_potential=None,
    **kwargs,
):
    r"""
    Helper function for building an instance of the :class:`deepinv.unfolded.BaseDEQ` class.

    .. note::

        .. note:: For now DEQ is only possible with PGD, HQS and GD optimization algorithms.

    :param str, deepinv.optim.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"PGD"`` (proximal gradient descent), ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford).
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
                            e.g. the stepsize, regularisation parameter, denoising standard deviation.
                            Each value of the dictionary can be either Iterable (distinct value for each iteration) or
                            a single float (same value for each iteration).
                            Default: ``{"stepsize": 1.0, "lambda": 1.0}``. See :any:`optim-params` for more details.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
                            Either a single instance (same data-fidelity for each iteration) or a list of instances of
                            :class:`deepinv.optim.DataFidelity` (distinct data-fidelity for each iteration). Default: ``None``.
    :param list, deepinv.optim.Prior prior: regularization prior.
                            Either a single instance (same prior for each iteration) or a list of instances of
                            deepinv.optim.Prior (distinct prior for each iteration). Default: ``None``.
    :param Callable F_fn: Custom user input cost function. default: None.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param deepinv.optim.Bregman bregman_potential: Bregman potential used for Bregman optimization algorithms such as Mirror Descent. Default: None, comes back to standart Euclidean optimization.
    :param kwargs: additional arguments to be passed to the :class:`deepinv.unfolded.BaseUnfold` class.
    """
    iterator = create_iterator(
        iteration,
        prior=prior,
        F_fn=F_fn,
        g_first=g_first,
        bregman_potential=bregman_potential,
    )
    return BaseDEQ(
        iterator,
        has_cost=iterator.has_cost,
        data_fidelity=data_fidelity,
        prior=prior,
        params_algo=params_algo,
        **kwargs,
    )
