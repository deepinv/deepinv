import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.optim_iterators import *
from deepinv.unfolded.unfolded import BaseUnfold
from deepinv.optim.optimizers import str_to_class
from deepinv.optim.data_fidelity import L2


class BaseDEQ(BaseUnfold):
    r"""
    Base class for deep equilibrium (DEQ) algorithms. Child of :class:`deepinv.unfolded.BaseUnfold`.

    Enables to turn any iterative optimization algorithm into a DEQ algorithm, i.e. an algorithm
    that can be virtually unrolled infinitely leveraging the implicit function theorem.
    The backward pass is performed using fixed point iterations to find solutions of the fixed-point equation

    .. math::
    \begin{equation}
    \label{eq:fixed_point}
    \tag{1}
    v = \left(\frac{\partial \operatorname{FixedPoint}(x^\star)}{\partial x^\star} \right )^T v + u.
    \end{equation}

    where :math:`u` is the incoming gradient from the backward pass, and :math:`x^\star` is the equilibrium point of the forward pass.

    See `<http://implicit-layers-tutorial.org/deep_equilibrium_models/>` for more details.

    :param int max_iter_backward: Maximum number of backward iterations. Default: 50.
    """

    def __init__(self, *args, max_iter_backward=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iter_backward = max_iter_backward

    def forward(self, y, physics, x_gt=None):
        r"""
        The forward pass of the DEQ algorithm. Compared to :class:`deepinv.unfolded.BaseUnfold`, the backward algorithm is performed using fixed point iterations.

        :param torch.Tensor y: Input tensor.
        :param deepinv.physics physics: Physics object.
        :return: Output torch.Tensor.
        """
        with torch.no_grad(): # Perform the forward pass without gradient tracking
            x, metrics = self.fixed_point(y, physics, x_gt=x_gt)
        # Once, at the equilibrium point, performs one additional iteration with gradient tracking
        cur_prior = self.update_prior_fn(self.max_iter-1) # prior should be constant over the iterations
        cur_params = self.update_params_fn(self.max_iter-1) # parameters should be constant over the iterations
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
                return {"est": (x0,), "cost": None} # initialize the fixed point algorithm.
            backward_FP = FixedPoint(
                backward_iterator(),
                init_iterate_fn=init_iterate_fn,
                max_iter=self.max_iter_backward,
                check_conv_fn=self.check_conv_fn
            )
            g = backward_FP({"est": (grad,)}, None)[0]["est"][0]
            return g
        if x.requires_grad:
            x.register_hook(backward_hook)
        if self.return_metrics:
            return x, metrics
        else:
            return x


def DEQ_builder(
    iteration,
    data_fidelity=L2(),
    F_fn=None,
    g_first=False,
    beta=1.0,
    max_iter_backward=50,
    **kwargs
):
    r"""
    Function building the appropriate Unfolded architecture.

    :param str, deepinv.optim.optim_iterators.OptimIterator iteration: either the name of the algorithm to be used, or an optim iterator .
        If an algorithm name (string), should be either `"PGD"`, `"ADMM"`, `"HQS"`, `"CP"` or `"DRS"`.
    :param deepinv.optim.DataFidelity data_fidelity: data fidelity term in the optimization problem.
    :param callable F_fn: Custom user input cost function. default: None.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param float beta: relaxation parameter in the fixed point algorithm. Default: `1.0`.
    :param int max_iter_backward: Maximum number of backward iterations. Default: 50.
    """
    # If no custom objective function F_fn is given but g is explicitly given, we have an explicit objective function.
    explicit_prior = (
        kwargs["prior"][0].explicit_prior
        if isinstance(kwargs["prior"], list)
        else kwargs["prior"].explicit_prior
    )
    if F_fn is None and explicit_prior:

        def F_fn(x, prior, cur_params, y, physics):
            return cur_params["lambda"] * data_fidelity(x, y, physics) + prior.g(
                x, cur_params["g_param"]
            )

        has_cost = True # boolean to indicate if there is a cost function to evaluate along the iterations
    else:
        has_cost = False
    # Create a instance of :class:`deepinv.optim.optim_iterators.OptimIterator`. 
    # If the iteration is directly given as an instance of OptimIterator, nothing to do
    if isinstance(iteration, str):
        iterator_fn = str_to_class(iteration + "Iteration") # If the name of the algorithm is given as a string, the correspondong class is automatically called. 
        iteration = iterator_fn(
            data_fidelity=data_fidelity,
            g_first=g_first,
            beta=beta,
            F_fn=F_fn,
            has_cost=has_cost,
        )
    return BaseDEQ(
        iteration, has_cost=has_cost, max_iter_backward=max_iter_backward, **kwargs
    )
