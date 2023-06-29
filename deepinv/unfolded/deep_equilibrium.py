import torch
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.optim_iterators import *
from deepinv.unfolded.unfolded import BaseUnfold
from deepinv.optim.optimizers import create_iterator
from deepinv.optim.data_fidelity import L2


class BaseDEQ(BaseUnfold):
    r"""
    Base class for deep equilibrium (DEQ) algorithms. Child of :class:`deepinv.unfolded.BaseUnfold`.

    Enables to turn any iterative optimization algorithm into a DEQ algorithm, i.e. an algorithm
    that can be virtually unrolled infinitely leveraging the implicit function theorem.
    The backward pass is performed using fixed point iterations to find solutions of the fixed-point equation

    .. math::

        \begin{equation}
        v = \left(\frac{\partial \operatorname{FixedPoint}(x^\star)}{\partial x^\star} \right )^T v + u.
        \end{equation}

    where :math:`u` is the incoming gradient from the backward pass,
    and :math:`x^\star` is the equilibrium point of the forward pass.

    See `this tutorial <http://implicit-layers-tutorial.org/deep_equilibrium_models/>`_ for more details.

    :param int max_iter_backward: Maximum number of backward iterations. Default: 50.
    """

    def __init__(self, *args, max_iter_backward=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iter_backward = max_iter_backward

    def forward(self, y, physics, x_gt=None, compute_metrics=False):
        r"""
        The forward pass of the DEQ algorithm. Compared to :class:`deepinv.unfolded.BaseUnfold`, the backward algorithm is performed using fixed point iterations.

        :param torch.Tensor y: Input tensor.
        :param deepinv.physics physics: Physics object.
        :param torch.Tensor x_gt: (optional) ground truth image, for plotting the PSNR across optim iterations.
        :param bool compute_metrics: whether to compute the metrics or not. Default: ``False``.
        :return: If ``compute_metrics`` is ``False``,  returns (:class:`torch.Tensor`) the output of the algorithm.
                Else, returns (:class:`torch.Tensor`, dict) the output of the algorithm and the metrics.
        """
        with torch.no_grad():  # Perform the forward pass without gradient tracking
            x, metrics = self.fixed_point(
                y, physics, x_gt=x_gt, compute_metrics=compute_metrics
            )
        # Once, at the equilibrium point, performs one additional iteration with gradient tracking.
        cur_prior = self.update_prior_fn(self.max_iter - 1)
        cur_params = self.update_params_fn(self.max_iter - 1)
        x = self.fixed_point.iterator(x, cur_prior, cur_params, y, physics)["est"][0]
        # Another iteration for jacobian computation via automatic differentiation.
        x0 = x.clone().detach().requires_grad_()
        f0 = self.fixed_point.iterator(
            {"est": (x0,)}, cur_prior, cur_params, y, physics
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

            backward_FP = FixedPoint(
                backward_iterator(),
                init_iterate_fn=init_iterate_fn,
                max_iter=self.max_iter_backward,
                check_conv_fn=self.check_conv_fn,
            )
            g = backward_FP({"est": (grad,)}, None)[0]["est"][0]
            return g

        if x.requires_grad:
            x.register_hook(backward_hook)

        if compute_metrics:
            return x, metrics
        else:
            return x


def DEQ_builder(iteration, params_algo={"lambda": 1.0, "stepsize": 1.0}, data_fidelity=L2(), F_fn=None, prior=None, g_first=False, **kwargs):
    r"""
    Helper function for building an Unfolded architecture.

    :param str, deepinv.optim.optim_iterators.OptimIterator iteration: either the name of the algorithm to be used, or an optim iterator .
        If an algorithm name (string), should be either `"PGD"`, `"ADMM"`, `"HQS"`, `"CP"` or `"DRS"`.
    :param deepinv.optim.DataFidelity data_fidelity: data fidelity term in the optimization problem.
    :param callable F_fn: Custom user input cost function. default: None.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param float beta: relaxation parameter in the fixed point algorithm. Default: `1.0`.
    """
    # If no custom objective function F_fn is given but g is explicitly given, we have an explicit objective function.

    iterator = create_iterator(iteration, data_fidelity=data_fidelity, prior=prior, F_fn=F_fn, g_first=g_first)
    return BaseDEQ(iterator, has_cost=iterator.has_cost, prior=prior, params_algo=params_algo, **kwargs)
