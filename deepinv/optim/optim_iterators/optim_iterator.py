import torch
import torch.nn as nn


class OptimIterator(nn.Module):
    r"""
    Base class for optimization iterators.

    An optim iterator is an object that implements a fixed point iteration for minimizing the sum of two functions
    :math:`F = f + \lambda \regname` where :math:`f` is a data-fidelity term  that will be modeled by an instance of physics
    and g is a regularizer. The fixed point iteration takes the form

    .. math::
        \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, \regname, A, y, ...)

    where :math:`x` is a "primal" variable converging to the solution of the minimization problem, and
    :math:`z` is a "dual" variable.


    .. note::
        By an abuse of terminology, we call "primal" and "dual" variables the variables that are updated
        at each step and which may correspond to the actual primal and dual variables from 
        (for instance in the case of the PD algorithm), but not necessarily (for instance in the case of the
        PGD algorithm).


    The implementation of the fixed point algorithm in :class:`deepinv.optim.FixedPoint` is split in two steps, alternating between
    a step on :math:`f` and a step on :math:`\regname`, that is for :math:`k=1,2,...`

    .. math::
        z_{k+1} = \operatorname{step}_f(x_k, z_k, y, A, ...)\\
        x_{k+1} = \operatorname{step}_{\regname}(x_k, z_k, y, A, ...)

    where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_{\regname}` are the steps on f and g respectively.

    :param bool g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param F_fn: function that returns the function F to be minimized at each iteration. Default: None.
    :param bool has_cost: If True, the function F is computed at each iteration. Default: False.
     """

    def __init__(self, g_first=False, F_fn=None, has_cost=False, **kwargs):
        super(OptimIterator, self).__init__()
        self.g_first = g_first
        self.F_fn = F_fn
        self.has_cost = has_cost
        if self.F_fn is None:
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

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        General form of a single iteration of splitting algorithms for minimizing :math:`F =  f + \lambda \regname`, alternating
        between a step on :math:`f` and a step on :math:`\regname`.
        The primal and dual variables as well as the estimated cost at the current iterate are stored in a dictionary
        `X` of the form `{'est': (x,z), 'cost': F}`.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"][0]
        if not self.g_first:
            z = self.f_step(
                x_prev, cur_data_fidelity, cur_params, y, physics, *args, **kwargs
            )
            x = self.g_step(z, cur_prior, cur_params, *args, **kwargs)
        else:
            z = self.g_step(x_prev, cur_prior, cur_params)
            x = self.f_step(
                z, cur_data_fidelity, cur_params, y, physics, *args, **kwargs
            )
        x = self.relaxation_step(x, x_prev, cur_params["beta"], *args, **kwargs)
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"est": (x, z), "cost": F}


class fStep(nn.Module):
    r"""
    Module for the single iteration steps on the data-fidelity term :math:`f`.

    :param bool g_first: If True, the algorithm starts with a step on g and finishes with a step on f. Default: False.
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self, g_first=False, **kwargs):
        super(fStep, self).__init__()
        self.g_first = g_first

        def forward(
            self, x, cur_data_fidelity, cur_params, y, physics, *args, **kwargs
        ):
            r"""
            Single iteration step on the data-fidelity term :math:`f`.

            :param torch.Tensor x: Current iterate.
            :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
            :param dict cur_params: Dictionary containing the current parameters of the algorithm.
            :param torch.Tensor y: Input data.
            :param deepinv.physics.Physics physics: Instance of the physics modeling the observation.
            """
            pass


class gStep(nn.Module):
    r"""
    Module for the single iteration steps on the prior term :math:`\lambda \regname`.

    :param bool g_first: If True, the algorithm starts with a step on g and finishes with a step on f. Default: False.
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self, g_first=False, **kwargs):
        super(gStep, self).__init__()
        self.g_first = g_first

        def forward(self, x, cur_prior, cur_params, *args, **kwargs):
            r"""
            Single iteration step on the prior term :math:`\regname`.

            :param torch.Tensor x: Current iterate.
            :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
            :param dict cur_params: Dictionary containing the current parameters of the algorithm.
            """
            pass
