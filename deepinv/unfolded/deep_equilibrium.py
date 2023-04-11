import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.unfolded.unfolded import BaseUnfold
from deepinv.optim.utils import str_to_class
from deepinv.optim.data_fidelity import L2

class BaseDEQ(BaseUnfold):

    def __init__(self, *args, max_iter_backward=50, **kwargs):
        super(BaseDEQ, self).__init__(*args, **kwargs)

        self.max_iter_backward = max_iter_backward

    def forward(self, y, physics):
        init_params = self.get_params_it(0)
        x = self.get_init(init_params, y, physics)
        with torch.no_grad():
            x, cur_params = self.fixed_point(x, init_params, y, physics, return_params=True)
        x = self.iterator(x, cur_params, y, physics)['est'][0]
        x0 = x.clone().detach().requires_grad_()
        f0 = self.iterator({'est': (x0,)} , cur_params, y, physics)['est'][0]

        def backward_hook(grad):
            iterator = lambda y, _ : {'est' : (torch.autograd.grad(f0, x0, y['est'][0], retain_graph=True)[0] + grad,)}
            if self.anderson_acceleration :
                backward_FP = AndersonAcceleration(iterator, max_iter=self.max_iter_backward, history_size=self.anderson_history_size, beta=self.anderson_beta,
                            early_stop=self.early_stop, crit_conv=self.crit_conv, verbose=self.verbose)
            else :
                backward_FP = FixedPoint(iterator, max_iter=self.max_iter_backward, early_stop=self.early_stop, crit_conv=self.crit_conv, verbose=self.verbose)
            g = backward_FP({'est' : (grad,)}, None)['est'][0]
            return g

        if x.requires_grad:
            x.register_hook(backward_hook)

        return x


def DEQ(algo_name, params_algo, data_fidelity=L2(), F_fn=None, device='cpu', g=None, prox_g=None,
            grad_g=None, g_first=False, stepsize_inter=1., max_iter_inter=50, tol_inter=1e-3, 
            beta=1., **kwargs):
    iterator_fn = str_to_class(algo_name + 'Iteration')
    iterator = iterator_fn(data_fidelity=data_fidelity, device=device, g=g, prox_g=prox_g,
                 grad_g=grad_g, g_first=g_first, stepsize_inter=stepsize_inter,
                 max_iter_inter=max_iter_inter, tol_inter=tol_inter, beta=beta)
    return BaseDEQ(iterator, params_algo = params_algo, F_fn = F_fn, **kwargs)