import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.optim.data_fidelity import L2
from deepinv.optim.utils import str_to_class
from deepinv.optim.optimizers import BaseOptim

class BaseUnfold(BaseOptim):
    '''
    Unfolded module
    '''

    def __init__(self, *args, trainable_params = [], custom_g_step=None, custom_f_step=None, device=torch.device('cpu'), **kwargs):
        super(BaseUnfold, self).__init__(*args, **kwargs)
        for param_key in trainable_params: 
            if param_key in self.params_algo.keys():
                param_value = self.params_algo[param_key]
                self.params_algo[param_key] = nn.ParameterList([nn.Parameter(torch.tensor(el).to(device)) for el in param_value])    

        if custom_g_step is not None:
            self.iterator.g_step = custom_g_step
        if custom_f_step is not None:
            self.iterator.f_step = custom_f_step

        print(self.parameters)

def Unfolded(algo_name, params_algo, data_fidelity=L2(), F_fn=None, device='cpu', g=None, prox_g=None,
            grad_g=None, g_first=False, stepsize_inter=1., max_iter_inter=50, tol_inter=1e-3, 
            beta=1., trainable_params=[], **kwargs):
    iterator_fn = str_to_class(algo_name + 'Iteration')
    iterator = iterator_fn(data_fidelity=data_fidelity, device=device, g=g, prox_g=prox_g,
                 grad_g=grad_g, g_first=g_first, stepsize_inter=stepsize_inter,
                 max_iter_inter=max_iter_inter, tol_inter=tol_inter, beta=beta)
    return BaseUnfold(iterator, params_algo = params_algo, trainable_params = trainable_params, F_fn = F_fn, **kwargs)