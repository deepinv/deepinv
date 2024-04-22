import torch
import torch.nn as nn
from deepinv.optim.data_fidelity import L2


class OptimIterator(nn.Module):
    r"""
    Base class for all :meth:`Optim` iterators.

    An optim iterator is an object that implements a fixed point iteration for minimizing the sum of two functions
    :math:`F = f + \lambda g` where :math:`f` is a data-fidelity term  that will be modeled by an instance of physics
    and g is a regularizer. The fixed point iteration takes the form

    .. math::
        \qquad x_{k+1} = \operatorname{FixedPoint}(x_k, f, g, A, y, ...)

    where :math:`x` is an iterated fixed-point variable. 

    .. note::
        The fixed-point iterate :math:`x` does not necessarily correspond to the minimizer of :math:`F`. 
        This is typically the save for Douglas-Rachford splitting, ADMM or primal-dual algorithms. 
        In order to get the curent estimate of the minimizer of :math:`F` from the current iterate, one can use the function :math:`get_estimate_from_iterate`.


    The implementation of the fixed point algorithm in :meth:`deepinv.optim`  is split in two steps, alternating between
    a step on f and a step on g, that is for :math:`k=1,2,...`

    .. math::
        z_{k+1} = \operatorname{step}_f(x_k, y, A, ...)\\
        x_{k+1} = \operatorname{step}_g(z_k, y, A, ...)

    where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are the steps on f and g respectively.

    :param bool g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param cost_fn: function that returns the function F to be minimized at each iteration. Default: None.
    :param bool has_cost: If True, the function F is computed at each iteration. Default: False.
     """

    def __init__(self, g_first=False, cost_fn=None, has_cost=False, **kwargs):
        super(OptimIterator, self).__init__()
        self.g_first = g_first
        self.cost_fn = cost_fn
        self.has_cost = has_cost
        if self.cost_fn is None:
            self.has_cost = False
        self.f_step = fStep(g_first=self.g_first)
        self.g_step = gStep(g_first=self.g_first)
        self.requires_grad_g = False
        self.requires_prox_g = False

    def relaxation_step(self, u, v, beta):
        r"""
        Performs a relaxation step of the form :math:`\beta u + (1-\beta) v`.

        :param torch.Tensor u: First tensor.
        :param torch.Tensor v: Second tensor.
        :param float beta: Relaxation parameter.
        :return: Relaxed tensor.
        """
        return beta * u + (1 - beta) * v

    def get_estimate_from_iterate(
        self, iterate, cur_data_fidelity, cur_prior, cur_params, y, physics
    ):
        """
        Get the minimizer of F from the fixed point iterate.

        :param torch.Tensor iterate: Fixed point variable iterated by the algorithm.
        :return: Minimizer of F.
        """
        return iterate

    def init_algo(self, y, physics):
        """
        Initialize the fixed-point algorithm by computing the initial iterate and estimate.
        By default, the first iterate is chosen as :math:`A^{\top}x`.

        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the observation.

        :return: Dictionary containing the initial iterate and initial estimate.
        """
        x = physics.A_adjoint(y)
        return {"iterate": x, "estimate": x}

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        General form of a single iteration of splitting algorithms for minimizing :math:`F =  f + \lambda g`, alternating
        between a step on :math:`f` and a step on :math:`g`.
        The fixed-point iterate, the current estimate as well as the estimated cost at the current iterate are stored in a dictionary
        $X$ of the form `{'iterate' : x,  'estimate': z , 'cost': F}`.
        The variable `iterate` can be either a :class:`torch.Tensor` or a tuple of :class:`torch.Tensor` elements.

        :param dict X: Dictionary containing the current iterate, current estimate and cost at the current estimate.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{'iterate' : x,  'estimate': z , 'cost': F}` containing the updated iterate, estimate and cost value.
        """
        x_prev = X["iterate"]
        if not self.g_first:
            z = self.f_step(x_prev, cur_data_fidelity, cur_params, y, physics)
            x = self.g_step(z, cur_prior, cur_params)
        else:
            z = self.g_step(x_prev, cur_prior, cur_params)
            x = self.f_step(z, cur_data_fidelity, cur_params, y, physics)
        x = self.relaxation_step(x, x_prev, cur_params["beta"])
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


class fStep(nn.Module):
    r"""
    Module for the single iteration steps on the data-fidelity term :math:`f`.

    :param bool g_first: If True, the algorithm starts with a step on g and finishes with a step on f. Default: False.
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self, g_first=False, **kwargs):
        super(fStep, self).__init__()
        self.g_first = g_first

        def forward(self, x, cur_data_fidelity, cur_params, y, physics):
            r"""
            Single iteration step on the data-fidelity term :math:`f`.

            :param torch.Tensor x: Current iterate.
            :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
            :param dict cur_params: Dictionary containing the current parameters of the algorithm.
            :param torch.Tensor y: Input data.
            :param deepinv.physics physics: Instance of the physics modeling the observation.
            """
            pass


class gStep(nn.Module):
    r"""
    Module for the single iteration steps on the prior term :math:` \lambda g`.

    :param bool g_first: If True, the algorithm starts with a step on g and finishes with a step on f. Default: False.
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self, g_first=False, **kwargs):
        super(gStep, self).__init__()
        self.g_first = g_first

        def forward(self, x, cur_prior, cur_params):
            r"""
            Single iteration step on the prior term :math:`g`.

            :param torch.Tensor x: Current iterate.
            :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
            :param dict cur_params: Dictionary containing the current parameters of the algorithm.
            """
            pass
