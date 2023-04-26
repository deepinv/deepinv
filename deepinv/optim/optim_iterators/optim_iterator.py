import torch
import torch.nn as nn
from deepinv.optim.data_fidelity import L2


class OptimIterator(nn.Module):
    r"""
    Base class for all :meth:`Optim` iterators.

    An optim iterator is an object that implements a fixed point iteration for minimizing the sum of two functions
    :math:`F = \lambda*f + g` where :math:`f` is a data-fidelity term  that will be modeled by an instance of physics
    and g is a regularizer. The fixed point iteration takes the form

    .. math::
        \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, g, A, y, ...)

    where :math:`x` is a "primal" variable converging to the solution of the minimisation problem, and
    :math:`z` is a "dual" variable.


    .. note::
        By an abuse of terminology, we call "primal" and "dual" variables the variables that are updated
        at each step and which may correspond to the actual primal and dual variables from optimisation algorithms
        (for instance in the case of the PD algorithm), but not necessarily (for instance in the case of the
        PGD algorithm).


    The implementation of the fixed point algorithm in :meth:`deepinv.optim`  is split in two steps, alternating between
    a step on f and a step on g, that is for :math:`k=1,2,...`

    .. math::
        z_{k+1} = \operatorname{step}_f(x_k, z_k, y, A, ...)\\
        x_{k+1} = \operatorname{step}_g(x_k, z_k, y, A, ...)

    where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are the steps on f and g respectively.

    :param data_fidelity: data_fidelity instance modeling the data-fidelity term.
    :param g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param float beta: relaxation parameter for the fixed-point iterations.
    :param F_fn: function that returns the function F to be minimized at each iteration. Default: None.
    :param str bregman_potential: Bregman potential to be used for the step on g. Default: "L2".
    """

    def __init__(
        self,
        data_fidelity=L2(),
        g_first=False,
        beta=1.0,
        F_fn=None,
        bregman_potential="L2",
    ):
        super(OptimIterator, self).__init__()
        self.data_fidelity = data_fidelity
        self.beta = beta
        self.g_first = g_first
        self.F_fn = F_fn
        self.bregman_potential = bregman_potential
        self.f_step = fStep(
            data_fidelity=self.data_fidelity,
            g_first=self.g_first,
            bregman_potential=self.bregman_potential,
        )
        self.g_step = gStep(
            g_first=self.g_first, bregman_potential=self.bregman_potential
        )

    def relaxation_step(self, u, v):
        r"""
        Performs a relaxation step of the form :math:`\beta u + (1-\beta) v`.

        :param torch.Tensor u: First tensor.
        :param torch.Tensor v: Second tensor.
        :return: Relaxed tensor.
        """
        return self.beta * u + (1 - self.beta) * v

    def forward(self, X, prior, cur_params, y, physics):
        r"""
        General form of a single iteration of splitting algorithms for minimizing :math:`F = \lambda f + g`, alternating
        between a step on :math:`f` and a step on :math:`g`.
        The primal and dual variables as well as the estimated cost at the current iterate are stored in a dictionary
        $X$ of the form `{'est': (x,z), 'cost': F}`.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param dict prior: dictionary containing the prior-related term of interest, e.g. its proximal operator or gradient.
        :param dict cur_params: dictionary containing the current parameters of the model.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"][0]
        if not self.g_first:
            z = self.f_step(x_prev, cur_params, y, physics)
            x = self.g_step(z, prior, cur_params)
        else:
            z = self.g_step(x_prev, prior, cur_params)
            x = self.f_step(z, cur_params, y, physics)
        x = self.relaxation_step(x, x_prev)
        F = self.F_fn(x, cur_params, y, physics) if self.F_fn else None
        return {"est": (x, z), "cost": F}


class fStep(nn.Module):
    r"""
    Module for the single iteration steps on the data-fidelity term :math:`f`.

    :param deepinv.optim.data_fidelity data_fidelity: data_fidelity instance modeling the data-fidelity term.
    :param bool g_first: If True, the algorithm starts with a step on g and finishes with a step on f. Default: False.
    :param str bregman_potential: Bregman potential to be used for the step on g. Default: "L2".
    :param kwargs: Additional keyword arguments.
    """

    def __init__(
        self, data_fidelity=L2(), g_first=False, bregman_potential="L2", **kwargs
    ):
        super(fStep, self).__init__()
        self.data_fidelity = data_fidelity
        self.g_first = g_first
        self.bregman_potential = bregman_potential

        def forward(self, x, cur_params, y, physics):
            r"""
            Single iteration step on the data-fidelity term :math:`f`.

            :param torch.Tensor x: Current iterate.
            :param dict cur_params: Dictionary containing the current fStep parameters (e.g. stepsizes).
            :param torch.Tensor y: Input data.
            :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
            """
            pass


class gStep(nn.Module):
    r"""
    Module for the single iteration steps on the prior term :math:`g`.

    :param bool g_first: If True, the algorithm starts with a step on g and finishes with a step on f. Default: False.
    :param str bregman_potential: Bregman potential to be used for the step on g. Default: "L2".
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self, g_first=False, bregman_potential="L2", **kwargs):
        super(gStep, self).__init__()
        self.g_first = g_first
        self.bregman_potential = bregman_potential

        def forward(self, x, cur_prior, cur_params):
            r"""
            Single iteration step on the prior term :math:`g`.

            :param torch.Tensor x: Current iterate.
            :param dict cur_prior: Dictionary containing the current prior.
            :param dict cur_params: Dictionary containing the current gStep parameters (e.g. stepsizes and regularisation parameters).
            """
            pass
