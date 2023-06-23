import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import BaseOptim, str_to_class


class BaseUnfold(BaseOptim):
    r"""
    Base class for unfolded algorithms.

    Enables to turn any proximal algorithm into an unfolded algorithm, i.e. an algorithm
    that can be trained end-to-end, with learnable parameters. Recall that the algorithms have the
    following form (see :meth:`deepinv.unfolded`):


    .. math::
        \begin{aligned}
        z_{k+1} &= \operatorname{step}_f(x_k, z_k, y, A, \lambda, \gamma, ...)\\
        x_{k+1} &= \operatorname{step}_g(x_k, z_k, y, A, \sigma, ...)
        \end{aligned}


    where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are learnable modules encompassing external
    parameters such as the measurement operator through the physics module :math:`A`, the data-fidelity term :math:`f`,
    a prior term :math:`g`, stepsizes...


    :param list trainable_params: List of parameters to be trained. Each parameter is a key of the `params_algo` dictionary for the :class:`deepinv.optim.optim_iterators.BaseIterator` class.
    :param callable custom_g_step: Custom gStep module. Default: None.
    :param callable custom_f_step: Custom fStep module. Default: None.
    :param torch.device device: Device on which to perform the computations. Default: `torch.device("cpu")`.
    :param kwargs: Keyword arguments to be passed to the :class:`deepinv.optim.optim_iterators.BaseIterator` class.
    """

    def __init__(
        self,
        iterator,
        trainable_params=[],
        custom_g_step=None,
        custom_f_step=None,
        device=torch.device("cpu"),
        **kwargs
    ):
        super(BaseUnfold, self).__init__(iterator, **kwargs)

        for param_key in trainable_params:
            if param_key in self.init_params_algo.keys():
                param_value = self.init_params_algo[param_key]
                self.init_params_algo[param_key] = nn.ParameterList(
                    [nn.Parameter(torch.tensor(el).to(device)) for el in param_value]
                )
        self.init_params_algo = nn.ParameterDict(self.init_params_algo)
        self.params_algo = self.init_params_algo.copy()
        self.prior = nn.ModuleList(self.prior)

        if custom_g_step is not None:
            self.iterator.g_step = custom_g_step
        if custom_f_step is not None:
            self.iterator.f_step = custom_f_step


def Unfolded(
    algo,
    trainable_params=[],
    data_fidelity=L2(),
    F_fn=None,
    g_first=False,
    beta=1.0,
    bregman_potential="L2",
    **kwargs
):
    r"""
    Function building the appropriate Unfolded architecture.

    :param algo: either name of the algorithm to be used, or an iterator. If an algorithm name (string), should be either `"PGD"`, `"ADMM"`, `"HQS"`, `"CP"` or `"DRS"`.
    :param deepinv.optim.DataFidelity data_fidelity: data fidelity term in the optimization problem.
    :param F_fn: Custom user input cost function. Default: None.
    :param g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. Default: False.
    :param float beta: relaxation parameter in the fixed point algorithm. Default: `1.0`.
    :param str bregman_potential: possibility to perform optimization with another bregman geometry. Default: `"L2"`
    """
    if isinstance(algo, str):
        iterator_fn = str_to_class(algo + "Iteration")
        iterator = iterator_fn(
            data_fidelity=data_fidelity,
            g_first=g_first,
            beta=beta,
            F_fn=F_fn,
            bregman_potential=bregman_potential,
        )
    else:
        iterator = algo
    kwargs["F_fn"] = F_fn
    return BaseUnfold(iterator, trainable_params=trainable_params, **kwargs)
