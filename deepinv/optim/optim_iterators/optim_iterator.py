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

    def __init__(self, data_fidelity=L2(), lamb=1., device='cpu', g=None, prox_g=None, grad_g=None, g_first=False, 
        stepsize_inter=1., max_iter_inter=50, tol_inter=1e-3, beta=1., F_fn = None):
        super(OptimIterator, self).__init__()
        self.data_fidelity = data_fidelity
        self.lamb = lamb
        self.beta = beta
        self.g_first = g_first
        self.g = g 
        self.F_fn = F_fn
        self.f_step = fStep(data_fidelity=self.data_fidelity, lamb=self.lamb, g_first=self.g_first)
        self.g_step = gStep(prox_g=prox_g, grad_g=grad_g, g_first=self.g_first,
                            max_iter_inter=max_iter_inter, tol_inter=tol_inter, stepsize_inter=stepsize_inter)
        
    def relaxation_step(self, u, v):
        return self.beta * u + (1 - self.beta) * v

    def forward(self, X, cur_params, y, physics):
        '''
        General form of a single iteration of splitting algorithms for minimizing $F = \lambda f + g$. Can be overwritten for specific other forms.
        $X$ is a dictionary of the form {'est': (x,z), 'cost': F} where $x$ and $z$ are respectively the primal and dual variables.
        '''
        x_prev = X['est'][0]
        if not self.g_first:
            z = self.f_step(x_prev, cur_params, y, physics)
            x = self.g_step(z, cur_params)
        else:
            z = self.g_step(x_prev, cur_params)
            x = self.f_step(z, cur_params, y, physics)
        x = self.relaxation_step(x, x_prev)
        F = self.F_fn(x,cur_params,y,physics) if self.F_fn else None
        return {'est': (x,z), 'cost': F}


class fStep(nn.Module):
    def __init__(self, data_fidelity=L2(), lamb=1., g_first=False, **kwargs):
        super(fStep, self).__init__()
        self.lamb = lamb
        self.data_fidelity = data_fidelity
        self.g_first = g_first

        def forward(self, x, cur_params, y, physics):
            pass

class gStep(nn.Module):
    def __init__(self,g=None, prox_g=None, grad_g=None, g_first=False, 
                    max_iter_inter=50, stepsize_inter=1., tol_inter=1e-3, **kwargs):
        super(gStep, self).__init__()
        self.g_first = g_first
        self.prox_g = prox_g
        self.grad_g = grad_g

        if prox_g is None and grad_g is None:
            if g is not None and isinstance(g, nn.Module):
                def grad_g(self, x, *args):
                    torch.set_grad_enabled(True)
                    return torch.autograd.grad(g(x, *args), x, create_graph=True, only_inputs=True)[0]
                from deepinv.optim.utils import gradient_descent
                def prox_g(self, x, *args):
                    grad = lambda y: grad_g(y, *args) + (1 / 2) * (y - x)
                    return gradient_descent(grad, x, stepsize_inter, max_iter=max_iter_inter, tol=tol_inter)
            else:
                raise ValueError('Either g is a nn.Module or prox_g and grad_g are provided.')

        def forward(self, x, cur_params):
            pass


