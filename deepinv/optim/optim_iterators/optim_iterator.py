import torch
import torch.nn as nn
from deepinv.optim.data_fidelity import L2

class OptimIterator(nn.Module):
    '''
    Optimization algorithms Fixed Point Iterations for minimizing the sum of two functions F = \lambda*f + g where f is a data-fidelity term that will me modeled by an instance of physics
    and g is a regularizer either explicitly or implicitly given by either its prox or its gradient.
    By default, the algorithms starts with a step on f and finishes with step on g.

    :param data_fidelity: data_fidelity instance modeling the data-fidelity term.
    :param lamb: Regularization parameter.
    :param g: Regularizing potential.
    :param prox_g: Proximal operator of the regularizing potential. x, g_param, it -> prox_g(x, g_param, it)
    :param grad_g: Gradient of the regularizing potential. x, g_param, it -> grad_g(x, g_param, it)
    :param g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param stepsize: Step size of the algorithm.
    '''

    def __init__(self, data_fidelity=L2(), g_first=False, beta=1., F_fn = None, bregman_potential='L2'):
        super(OptimIterator, self).__init__()
        self.data_fidelity = data_fidelity
        self.beta = beta
        self.g_first = g_first
        self.F_fn = F_fn
        self.bregman_potential = bregman_potential
        self.f_step = fStep(data_fidelity=self.data_fidelity, g_first=self.g_first, bregman_potential=self.bregman_potential)
        self.g_step = gStep(g_first=self.g_first, bregman_potential=self.bregman_potential)
        
    def relaxation_step(self, u, v):
        return self.beta * u + (1 - self.beta) * v

    def forward(self, X, prior, cur_params, y, physics):
        '''
        General form of a single iteration of splitting algorithms for minimizing $F = \lambda f + g$. Can be overwritten for specific other forms.
        $X$ is a dictionary of the form {'est': (x,z), 'cost': F} where $x$ and $z$ are respectively the primal and dual variables.
        '''
        x_prev = X['est'][0]
        if not self.g_first:
            z = self.f_step(x_prev, cur_params, y, physics)
            x = self.g_step(z, prior, cur_params)
        else:
            z = self.g_step(x_prev, prior, cur_params)
            x = self.f_step(z, cur_params, y, physics)
        x = self.relaxation_step(x, x_prev)
        F = self.F_fn(x,cur_params,y,physics) if self.F_fn else None
        return {'est': (x,z), 'cost': F}


class fStep(nn.Module):
    def __init__(self, data_fidelity=L2(), g_first=False, bregman_potential='L2', **kwargs):
        super(fStep, self).__init__()
        self.data_fidelity = data_fidelity
        self.g_first = g_first
        self.bregman_potential = bregman_potential

        def forward(self, x, cur_params, y, physics):
            pass

class gStep(nn.Module):
    def __init__(self, g_first=False, bregman_potential='L2', **kwargs):
        super(gStep, self).__init__()
        self.g_first = g_first
        self.bregman_potential = bregman_potential

        def forward(self, x, prior, cur_params):
            pass


