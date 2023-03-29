import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration
from deepinv.optim.optim_iterators import *
from deepinv.unfolded.unfolded import BaseUnfold
from deepinv.optim.utils import str_to_class

class BaseDEQ(BaseUnfold):
    '''
    Deep Equilibrium Model. Strongly inspired from http://implicit-layers-tutorial.org/deep_equilibrium_models/. 
    '''
    def __init__(self, *args, max_iter_backward=50, **kwargs):
        super(BaseDEQ, self).__init__(*args, **kwargs)

        self.max_iter_backward = max_iter_backward

    def forward(self, y, physics, **kwargs):
        x = self.get_init(y, physics)
        with torch.no_grad():
            x = self.fixed_point(x, y, physics, **kwargs)[0]
        x = self.iterator(x, 0, y, physics)[0]
        x0 = x.clone().detach().requires_grad_()
        f0 = self.iterator(x0, 0, y, physics)[0]

        def backward_hook(grad):
            grad = (grad,)
            iterator = lambda y,it: (torch.autograd.grad(f0, x0, y, retain_graph=True)[0] + grad[0],)
            if self.anderson_acceleration :
                backward_FP = AndersonAcceleration(iterator, max_iter=self.max_iter_backward, history_size=self.anderson_history_size, beta=self.anderson_beta,
                            early_stop=self.early_stop, crit_conv=self.crit_conv, verbose=self.verbose)
            else :
                backward_FP = FixedPoint(iterator, max_iter=self.max_iter_backward, early_stop=self.early_stop, crit_conv=self.crit_conv, verbose=self.verbose)
            g = backward_FP(grad)[0]
            return g

        if x.requires_grad:
            x.register_hook(backward_hook)
        return x


def DEQ(algo_name, data_fidelity='L2', lamb=1., device='cpu', g=None, prox_g=None,
                 grad_g=None, g_first=False, stepsize=[1.] * 50, g_param=None, stepsize_inter=1.,
                 max_iter_inter=50, tol_inter=1e-3, beta=1., **kwargs):
    iterator_fn = str_to_class(algo_name + 'Iteration')
    iterator = iterator_fn(data_fidelity=data_fidelity, lamb=lamb, device=device, g=g, prox_g=prox_g,
                 grad_g=grad_g, g_first=g_first, stepsize=stepsize, g_param=g_param, stepsize_inter=stepsize_inter,
                 max_iter_inter=max_iter_inter, tol_inter=tol_inter, beta=beta)
    return BaseDEQ(iterator, **kwargs)