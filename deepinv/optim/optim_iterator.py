import torch
import torch.nn as nn
from deepinv.optim.utils import gradient_descent

class OptimIterator(nn.Module):
    '''
    Optimization algorithms Fixed Point Iterations for minimizing the sum of two functions \lambda*f + g where f is a data-fidelity term that will me modeled by an instance of physics
    and g is a regularizer either explicit or implicitly given by either its prox or its gradient. 
    By default, the algorithms starts with a step on f and finishes with step on g. 

    :param data_fidelity: data_fidelity instance modeling the data-fidelity term.   
    :param lamb: Regularization parameter.
    :param g: Regularizing potential. 
    :param prox_g: Proximal operator of the regularizing potential. x,it -> prox_g(x,it)
    :param grad_g: Gradient of the regularizing potential. x,it -> grad_g(x,it)
    :param g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param stepsize: Step size of the algorithm.
    '''

    def __init__(self, data_fidelity='L2', lamb=1., device='cpu', g = None, prox_g = None,
                 grad_g = None, g_first = False, stepsize=1., stepsize_inter = 1., max_iter_inter=50, 
                 tol_inter=1e-3, update_stepsize=None) :
        super().__init__()

        self.data_fidelity = data_fidelity
        self.lamb = lamb
        self.prox_g = prox_g
        self.grad_g = grad_g
        self.g_first = g_first
        self.device = device

        self.stepsize = lambda it : update_stepsize(it) if update_stepsize else stepsize

        if prox_g is None and grad_g is None and not trainable:
            if g is not None and isinstance(g, nn.Module):
                def grad_g(self,x,*args):
                    torch.set_grad_enabled(True)
                    return torch.autograd.grad(g(x,*args), x, create_graph=True, only_inputs=True)[0]
                def prox_g(self,x,*args) :
                    grad = lambda  y : grad_g(y,*args) + (1/2)*(y-x)
                    return gradient_descent(grad, x, stepsize_inter, max_iter=max_iter_inter, tol=tol_inter)
            else:
                raise ValueError('Either g is a nn.Module or prox_g and grad_g are provided.')
        
        def forward(self, x, it, y, physics):
            pass


class GD(OptimIterator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, it, y, physics):
        return x - self.stepsize(it)*(self.lamb*self.data_fidelity.grad(x, y, physics) + self.grad_g(x,it))


class HQS(OptimIterator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if primal_prox is not None:
            self._primal_prox = primal_prox
        else:
            self._primal_prox = self.primal_prox

        if dual_prox is not None:
            self._dual_prox = dual_prox
        else:
            self._dual_prox = self.dual_prox

    def primal_prox(self, x, y, physics, it):
        return self.data_fidelity.prox(x, y, physics, self.lamb*self.stepsize(it))

    def dual_prox(self, z, it):
        return self.prox_g(z,it)
    
    def forward(self, x, it, y, physics):
        if not self.g_first:
            z = self._primal_prox(x, y, physics, it)
            x = self._dual_prox(z, it)
        else:
            z = self._dual_prox(x, it)
            x = self._primal_prox(z, y, physics, it)
        return x

class PGD(OptimIterator):

    def __init__(self, primal_prox=None, dual_prox=None, **kwargs):
        super().__init__(**kwargs)

        if primal_prox is not None:
            self._primal_prox = primal_prox
        else:
            self._primal_prox = self.primal_prox

        if dual_prox is not None:
            self._dual_prox = dual_prox
        else:
            self._dual_prox = self.dual_prox

    def primal_prox(self, x, grad, it):
        return x - self.stepsize(it) * self.lamb * grad

    def dual_prox(self, x, it):
        return self.prox_g(x, it)

    def forward(self, x, it, y, physics):
        if not self.g_first: # prox on g and grad on f
            grad = self.data_fidelity.grad(x, y, physics)
            z = self._primal_prox(x, grad, it)
            x = self._dual_prox(z, it)
        else:  # TODO: refactor  # prox on f and grad on g
            z = x - self.stepsize(it)*self.grad_g(x)
            x = self.data_fidelity.prox(z, y, physics, self.lamb*self.stepsize(it))
        return x

class DRS(OptimIterator):

    def __init__(self, primal_prox=None, dual_prox=None, **kwargs):
        super().__init__(**kwargs)

        if primal_prox is not None:
            self._primal_prox = primal_prox
        else:
            self._primal_prox = self.primal_prox

        if dual_prox is not None:
            self._dual_prox = dual_prox
        else:
            self._dual_prox = self.dual_prox

    def primal_prox(self, x, y, physics, it):
        return self.data_fidelity.prox(x, y, physics, self.lamb*self.stepsize(it))

    def dual_prox(self, z, it):
        return self.prox_g(z, it)
    
    def forward(self, x, it, y, physics):
        if not self.g_first:
            Rprox_f = 2*self._primal_prox(x, y, physics, it)-x
            Rprox_g = 2*self._dual_prox(Rprox_f, it)-Rprox_f
            x = (1/2)*(x + Rprox_g)
        else:
            Rprox_g = 2*self._dual_prox(x, it)-x
            Rprox_f = 2*self._primal_prox(Rprox_g, y, physics, it) - Rprox_g
            x = (1/2)*(x + Rprox_f)
        return x

class ADMM(OptimIterator):

    def __init__(self, theta, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, it, y, physics):
        # TODO : same as DRS ???
        pass


class PD(OptimIterator):

    def __init__(self, data_fidelity, update_stepsize=None, stepsize_2=1.,
                 primal_prox=None, dual_prox=None, **kwargs):
        '''
        In this case the algorithm works on the product space HxH^* so input/output variable is a concatenation of
        primal and dual variables.

        TODO:
        - check that there is no conflict with the data_fidelity.prox
        - check that there is freedom in how to apply replacement of prox operators (see J. Adler works)
        '''
        super(PD, self).__init__(**kwargs)

        self.stepsize_2 = lambda it : update_stepsize(it) if update_stepsize else stepsize_2

        self.data_fidelity = data_fidelity

        if primal_prox is not None:
            self._primal_prox = primal_prox
        else:
            self._primal_prox = self.primal_prox

        if dual_prox is not None:
            self._dual_prox = dual_prox
        else:
            self._dual_prox = self.dual_prox

    def primal_prox(self, x, Atu, y, it):
        return self.prox_g(x - self.stepsize_2(it) * Atu, it)

    def dual_prox(self, Ax_cur, u, y, it):
        v = u + self.stepsize(it) / 2. * Ax_cur
        return v - self.stepsize(it) / 2. * self.data_fidelity.prox_norm(v / (self.stepsize(it) / 2.), y, self.lamb)


    def forward(self, pd_var, it, y, physics):

        x, u = pd_var

        x_ = self._primal_prox(x, physics.A_adjoint(u), y, it)
        Ax_cur = physics.A(2*x_ - x)
        u = self._dual_prox(Ax_cur, u, y, it)

        pd_variable = (x_, u)

        return pd_variable
        

# def ADMM(self, y, physics, init=None):
#         '''
#         Alternating Direction Method of Multipliers (ADMM)
#         :param y: Degraded image.
#         :param physics: Physics instance modeling the degradation.
#         :param init: Initialization of the algorithm. If None, the algorithm starts from y.
#         '''
#         if init is None:
#             w = y
#         else:
#             w = init
#         x = torch.zeros_like(w)
#         for it in range(self.max_iter):
#             x_prev = x
#             if not self.g_first :
#                 z = self.data_fidelity.prox(w-x, y, physics, self.lamb*self.stepsize[it])
#                 w = self.prox_g(z+x_prev, it)
#             else :
#                 z = self.prox_g(w-x, it)
#                 w = self.data_fidelity.prox(z+x_prev, y, physics, self.lamb*self.stepsize[it])
#             x = x_prev + self.theta[it]*(z - w)
#             if not self.unroll and check_conv(x_prev,x,it, self.crit_conv, self.verbose) :
#                 break
#         return w
