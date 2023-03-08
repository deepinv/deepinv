import torch
import torch.nn as nn
from deepinv.optim.utils import gradient_descent
from .steps import PDFStep, PDGStep


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
                 grad_g=None, g_first=False, stepsize=[1.] * 50, g_param=[1.] * 50, stepsize_inter=1.,
                 max_iter_inter=50,
                 tol_inter=1e-3, beta=1.,
                 f_step=None, g_step=None):
        super(OptimIterator, self).__init__()

        self.data_fidelity = data_fidelity
        self.lamb = lamb
        self.prox_g = prox_g
        self.grad_g = grad_g
        self.g_first = g_first
        self.device = device
        self.stepsize = stepsize
        self.g_param = g_param
        self.beta = beta

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

            # self._f_step = f_step
            # self._g_step = g_step

    # def g_step(self, *args):
    #     return self._g_step(*args)
    #
    # def f_step(self, *args):
    #     return self._f_step(*args)
    # def g_step(self, x, it):
    #     pass
    #
    # def f_step(self, y, physics, it):
    #     pass

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
        return (x,)


class GD(OptimIterator):  # TODO

    def __init__(self, **kwargs):
        super(GD, self).__init__(**kwargs)

    def forward(self, x, it, y, physics):
        x = x[0]
        x = x - self.stepsize[it] * (
                    self.lamb * self.data_fidelity.grad(x, y, physics) + self.grad_g(x, g_param[it], it))
        return (x,)


class HQS(OptimIterator):

    def __init__(self, **kwargs):
        super(HQS, self).__init__(**kwargs)

    def f_step(self, x, y, physics, it):
        return self.data_fidelity.prox(x, y, physics, self.lamb * self.stepsize[it])

    def g_step(self, z, it):
        return self.prox_g(z, self.g_param[it], it)


class PGD(OptimIterator):

    def __init__(self, **kwargs):
        super(PGD, self).__init__(**kwargs)

    def f_step(self, x, y, physics, it):
        if not self.g_first:
            return x - self.stepsize[it] * self.lamb * self.data_fidelity.grad(x, y, physics)
        else:
            return self.data_fidelity.prox(x, y, physics, self.lamb * self.stepsize[it])

    def g_step(self, x, it):
        if not self.g_first:
            return self.prox_g(x, self.g_param[it], it)
        else:
            return x - self.stepsize[it] * self.grad_g(x, self.g_param[it], it)


class DRS(OptimIterator):

    def __init__(self, beta=0.5, **kwargs):
        super(DRS, self).__init__(**kwargs)
        self.beta = beta

    def f_step(self, x, y, physics, it):
        return 2 * self.data_fidelity.prox(x, y, physics, self.lamb * self.stepsize[it]) - x

    def g_step(self, z, it):
        return 2 * self.prox_g(z, self.g_param[it], it) - z


class ADMM(OptimIterator):

    def __init__(self, theta, **kwargs):
        super(ADMM, self).__init__(**kwargs)

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

        self.stepsize_2 = [stepsize_2/2.]*len(self.stepsize)
        self.data_fidelity = data_fidelity
        # self.g_step = PDGStep(prox_g=self.prox_g, g_param=self.g_param, stepsize_2=self.stepsize_2)
        # self.f_step = PDFStep(data_fidelity=self.data_fidelity, stepsize=self.stepsize, lamb=self.lamb)

    def _g_step(self, x, Atu, it):
        return self.prox_g(x - self.stepsize_2[it] * Atu, self.stepsize_2[it] * self.g_param[it], it)

    def _f_step(self, Ax_cur, u, y,
                  it):  # Beware this is not the prox of f(A\cdot) but only the prox of f, A is tackled independently in PD
        v = u + self.stepsize[it] * Ax_cur
        return v - self.stepsize[it] * self.data_fidelity.prox_norm(v / self.stepsize[it], y, self.lamb)

    def forward(self, pd_var, it, y, physics):

        x, u = pd_var

        x_ = self.g_step(x, physics.A_adjoint(u), it)
        Ax_cur = physics.A(2 * x_ - x)
        u_ = self.f_step(Ax_cur, u, y, it)

        pd_variable = (x_, u_)

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
