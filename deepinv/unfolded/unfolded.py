import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.optim.data_fidelity import L2
from deepinv.optim.utils import str_to_class
from deepinv.optim.optimizers import BaseOptim


class BaseUnfold(BaseOptim):
    """
    Unfolded module
    """

    def __init__(
        self,
        *args,
        trainable_params=[],
        custom_g_step=None,
        custom_f_step=None,
        device=torch.device("cpu"),
        **kwargs
    ):
        super(BaseUnfold, self).__init__(*args, **kwargs)
        for param_key in trainable_params:
            if param_key in self.params_algo.keys():
                param_value = self.params_algo[param_key]
                self.params_algo[param_key] = nn.ParameterList(
                    [nn.Parameter(torch.tensor(el).to(device)) for el in param_value]
                )
        self.params_algo = nn.ParameterDict(self.params_algo)

        for key, value in zip(self.prior.keys(), self.prior.values()):
            self.prior[key] = nn.ModuleList(value)
        self.prior = nn.ModuleDict(self.prior)

        if custom_g_step is not None:
            self.iterator.g_step = custom_g_step
        if custom_f_step is not None:
            self.iterator.f_step = custom_f_step


def Unfolded(
    algo_name,
    data_fidelity=L2(),
    F_fn=None,
    g_first=False,
    beta=1.0,
    bregman_potential="L2",
    **kwargs
):
    iterator_fn = str_to_class(algo_name + "Iteration")
    iterator = iterator_fn(
        data_fidelity=data_fidelity,
        g_first=g_first,
        beta=beta,
        F_fn=F_fn,
        bregman_potential=bregman_potential,
    )
    return BaseUnfold(
        iterator,
        F_fn=F_fn,
        **kwargs
    )
