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

        if g is not None and isinstance(g, nn.Module) and prox_g is None and grad_g is None:
            def grad_g(self,x,*args):
                torch.set_grad_enabled(True)
                return torch.autograd.grad(g(x,*args), x, create_graph=True, only_inputs=True)[0]
            def prox_g(self,x,*args) :
                grad = lambda  y : grad_g(y,*args) + (1/2)*(y-x)
                return gradient_descent(grad, x, stepsize_inter, max_iter=max_iter_inter, tol=tol_inter)
        else :
            raise ValueError
        
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
    
    def forward(self, x, it, y, physics):
        if not self.g_first : 
            z = self.data_fidelity.prox(x, y, physics, self.lamb*self.stepsize(it))
            x = self.prox_g(z,it)
        else :
            z = self.prox_g(z,it)
            x = self.data_fidelity.prox(z, y, physics, self.lamb*self.stepsize(it))
        return x

class PGD(OptimIterator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, it, y, physics):
        if not self.g_first : # prox on g and grad on f
            z = x - self.stepsize(it)*self.lamb*self.data_fidelity.grad(x, y, physics)
            x = self.prox_g(z,it)
        else :  # prox on f and grad on g
            z = x - self.stepsize(it)*self.grad_g(x,it)
            x = self.data_fidelity.prox(z, y, physics, self.lamb*self.stepsize(it))
        return x

class DRS(OptimIterator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, it, y, physics):
        if not self.g_first :
            Rprox_f = 2*self.data_fidelity.prox(x, y, physics, self.lamb*self.stepsize(it))-x
            Rprox_g = 2*self.prox_g(Rprox_f,it)-Rprox_f
            x = (1/2)*(x + Rprox_g)
        else :
            Rprox_g = 2*self.prox_g(x,it)-x
            Rprox_f = 2*self.data_fidelity.prox(Rprox_g, y, physics, self.lamb*self.stepsize(it))-Rprox_g
            x = (1/2)*(x + Rprox_g)
        return x

class ADMM(OptimIterator):

    def __init__(self, theta, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, it, y, physics):
        # TODO : same as DRS ???
        pass


class PD(OptimIterator):

    def __init__(self, **kwargs):
        '''
        In this case the algorithm works on the product space HxH^* so input/output variable is a concatenation of
        primal and dual variables.

        TODO:
        - check that there is no conflict with the data_fidelity.prox
        - check that there is
        '''
        super().__init__(**kwargs)

    def forward(self, pd_var, it, y, physics):

        x, u = pd_var[:, :pd_var.shape[1]//2, ...], pd_var[:, pd_var.shape[1]//2:, ...]

        x_ = self.prox_g(x - gamma * physics.A_adjoint(u), it)
        v = u + sigma * physics.A(2 * x_ - x)
        u = v - sigma * self.data_fidelity.prox(v / sigma, y, physics, self.lamb * self.stepsize(it) / sigma)

        pd_variable = torch.cat((x_, u), axis=1)

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
