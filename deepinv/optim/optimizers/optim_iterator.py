import torch
import torch.nn as nn
from deepinv.optim.utils import gradient_descent


class fStep(nn.Module):
    def __init__(self, data_fidelity='L2', lamb=1., g_first=False, stepsize=[1.] * 50, f_step=None, **kwargs):
        super(fStep, self).__init__()
        self.stepsize = stepsize
        self.lamb = lamb
        self.data_fidelity = data_fidelity
        self.g_first = g_first

        def forward(self, x, y, physics, it):
            pass

class gStep(nn.Module):
    def __init__(self,g=None, prox_g=None, grad_g=None, g_param=None, stepsize=[1.] * 50, g_first=False, max_iter_inter=50,
                 tol_inter=1e-3, g_step=None, **kwargs):
        super(gStep, self).__init__()
        self.stepsize = stepsize
        self.g_first = g_first
        self.prox_g = prox_g
        self.grad_g = grad_g
        self.g_param = g_param

        if prox_g is None and grad_g is None:
            if g is not None and isinstance(g, nn.Module):
                def grad_g(self, x, *args):
                    torch.set_grad_enabled(True)
                    return torch.autograd.grad(g(x, *args), x, create_graph=True, only_inputs=True)[0]

                def prox_g(self, x, *args):
                    grad = lambda y: grad_g(y, *args) + (1 / 2) * (y - x)
                    return gradient_descent(grad, x, stepsize_inter, max_iter=max_iter_inter, tol=tol_inter)
            else:
                raise ValueError('Either g is a nn.Module or prox_g and grad_g are provided.')

        def forward(self, x, it):
            pass

class OptimIterator(nn.Module):
    '''
    Optimization algorithms Fixed Point Iterations for minimizing the sum of two functions \lambda*f + g where f is a data-fidelity term that will me modeled by an instance of physics
    and g is a regularizer either explicit or implicitly given by either its prox or its gradient.
    By default, the algorithms starts with a step on f and finishes with step on g.

    TODO : adapt PD to the new g_step / f_step stype.
    TODO : update stepize PD removed.
    TODO : add accelerated algorithms.
    TODO : ADMM

    :param data_fidelity: data_fidelity instance modeling the data-fidelity term.
    :param lamb: Regularization parameter.
    :param g: Regularizing potential.
    :param prox_g: Proximal operator of the regularizing potential. x, g_param, it -> prox_g(x, g_param, it)
    :param grad_g: Gradient of the regularizing potential. x, g_param, it -> grad_g(x, g_param, it)
    :param g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param stepsize: Step size of the algorithm.
    '''

    def __init__(self, data_fidelity='L2', lamb=1., device='cpu', g=None, prox_g=None,
                 grad_g=None, g_first=False, stepsize=[1.] * 50, g_param=None, stepsize_inter=1.,
                 max_iter_inter=50, tol_inter=1e-3, beta=1., f_step=None, g_step=None):
        super(OptimIterator, self).__init__()

        self.f_step = fStep(data_fidelity=data_fidelity, lamb=lamb, g_first=g_first, stepsize=stepsize, f_step=f_step)
        self.g_step = gStep(prox_g=prox_g, grad_g=grad_g, g_param=g_param, stepsize=stepsize, g_first=g_first, max_iter_inter=max_iter_inter,
                 tol_inter=tol_inter, g_step=g_step)
        self.beta = beta
        self.g_first = g_first

    def relaxation_step(self, u, v):
        return self.beta * u + (1 - self.beta) * v

    def forward(self, x, it, y, physics):
        '''
        General splitting algorithm for minimizing \lambda f + g. Can be overwritten for specific other forms.
        Returns primal and dual updates.
        '''
        x_prev = x[0]
        if not self.g_first:
            x = self.f_step(x_prev, y, physics, it)
            x = self.g_step(x, it)
        else:
            x = self.g_step(x_prev, it)
            x = self.f_step(x, y, physics, it)
        x = self.relaxation_step(x, x_prev)
        return (x, )
