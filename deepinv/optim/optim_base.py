import torch
import torch.nn as nn


def check_conv(x_prev,x,it,crit_conv,verbose=False):
    crit_cur = (x_prev-x).norm() / (x.norm()+1e-03)
    if verbose:
        print(it, 'crit = ', crit_cur , '\r')
    if crit_conv is not None and crit_cur < crit_conv:
        has_converged = True
        return True
    else:
        return False


def conjugate_gradient(A, b, max_iter=1e2, tol=1e-5):
    '''
    Standard conjugate gradient algorithm to solve Ax=b
        see: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    :param A: Linear operator as a callable function, has to be square!
    :param b: input tensor
    :param max_iter: maximum number of CG iterations
    :param tol: absolute tolerance for stopping the CG algorithm.
    :return: torch tensor x verifying Ax=b
    '''

    def dot(s1, s2):
        return (s1 * s2).flatten().sum()

    x = torch.zeros_like(b)

    r = b
    p = r
    rsold = dot(r, r)

    for i in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = dot(r, r)
        #print(rsnew.sqrt())
        if rsnew.sqrt() < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def gradient_descent(grad_f, x, step_size=1., max_iter=1e2, tol=1e-5):
    '''
    Standard gradient descent algorithm to solve min_x f(x)
    :param grad_f: gradient of function to bz minimized as a callable function.
    :param x: input tensor
    :param step_size: step size of the gradient descent algorithm.
    :param max_iter: maximum number of iterations
    :param tol: absolute tolerance for stopping the algorithm.
    :return: torch tensor x verifying min_x f(x)
    '''

    for i in range(int(max_iter)):
        x_prev = x
        x = x - step_size * grad_f(x)
        if check_conv(x_prev, x, i, crit_conv=tol) :
            break
    return x

class ProxOptim(nn.Module):
    '''
    Proximal Optimization algorithms for minimizing the sum of two functions \lambda*f + g where f is a data-fidelity term that will me modeled by an instance of physics
    and g is a regularizer either explicit or implicitly given by either its prox or its gradient. 
    By default, the algorithms starts with a step on f and finishes with step on g. 

    :param algo_name: Name of the optimization algorithm.
    :param data_fidelity: data_fidelity instance modeling the data-fidelity term.   
    :param lamb: Regularization parameter.
    :param g: Regularizing potential. 
    :param prox_g: Proximal operator of the regularizing potential. x,it -> prox_g(x,it)
    :param grad_g: Gradient of the regularizing potential. x,it -> grad_g(x,it)
    :param max_iter: Number of iterations.
    :param step_size: Step size of the algorithm. List or int. If list, the length of the list must be equal to max_iter.
    :param theta: Relacation parameter of the ADMM/DRS/PD algorithms.
    :param g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param crit_conv: Mimimum relative change in the solution to stop the algorithm.
    :param unroll: If True, the algorithm is unrolled in time.
    :param verbose: prints progress of the algorithm.
    '''

    def __init__(self, algo_name='PGD', data_fidelity='L2', lamb=1., device='cpu', g = None, prox_g = None,
                 grad_g = None, max_iter=10, stepsize=1., theta=1., g_first = False, crit_conv=None, unroll=False,
                 verbose=False, stepsize_inter = 1., max_iter_inter=50, tol_inter=1e-3) :
        super().__init__()

        self.algo_name = algo_name
        self.data_fidelity = data_fidelity
        self.lamb = lamb
        self.prox_g = prox_g
        self.grad_g = grad_g
        self.g_first = g_first
        self.unroll = unroll
        self.max_iter = max_iter
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.device = device
        self.has_converged = False

        if algo_name == 'GD' or ( algo_name == 'PGD' and self.g_first ) :
            requires_grad_g = True
            requires_prox_g = False
        else :
            requires_grad_g = False
            requires_prox_g = True

        if requires_grad_g and grad_g is None :
            if g is not None and isinstance(g, nn.Module) :
                torch.set_grad_enabled(True)
                self.grad_g = lambda x,it : torch.autograd.grad(g(x), x, create_graph=True, only_inputs=True)[0]
            else :
                raise ValueError('grad_g or nn.Module g must be provided for {}'.format(algo_name))

        if requires_prox_g and prox_g is None :
            if g is not None and isinstance(g, nn.Module) :
                torch.set_grad_enabled(True)
                grad_g = lambda x,it : torch.autograd.grad(g(x), x,create_graph=True, only_inputs=True)[0]
            if grad_g is not None :
                def prox_g(self,x,it) :
                    grad_f = lambda  y : grad_g(y,it) + (1/2)*(y-x)
                    return gradient_descent(grad_f, x, stepsize_inter, max_iter=max_iter_inter, tol=tol_inter)
            else :
                raise ValueError('prox_g, grad_g or nn.Module g must be provided for {}'.format(algo_name))


        if isinstance(stepsize, float):
            self.stepsize = [stepsize] * max_iter
        elif isinstance(stepsize, list):
            assert len(stepsize) == max_iter
            self.stepsize = stepsize
        else:
            raise ValueError('stepsize must be either int/float or a list of length max_iter') 
        
        if self.unroll : 
            self.register_parameter(name='stepsize',
                                param=torch.nn.Parameter(torch.tensor(stepsize, device=self.device),
                                requires_grad=True))
        
        if isinstance(stepsize, float):
            self.theta = [theta] * max_iter
        elif isinstance(theta, list):
            assert len(theta) == max_iter
            self.theta = theta
        else:
            raise ValueError('stepsize must be either int/float or a list of length max_iter') 

    def GD(self, y, physics, init=None) : 
        '''
        Gradient Descent (GD)

        :param y: Degraded image.
        :param physics: Physics instance modeling the degradation.
        :param init: Initialization of the algorithm. If None, the algorithm starts from y.
        '''
        if init is None:
            x = y
        else:
            x = init
        for it in range(self.max_iter):
            x_prev = x
            x - self.stepsize[it]*(self.lamb*self.data_fidelity.grad(x, y, physics) + self.grad_g(x,it))
            if not self.unroll and check_conv(x_prev,x,it, self.crit_conv, self.verbose) :
                break
        return x 


    def HQS(self, y, physics, init=None) : 
        '''
        Half Quadratric Splitting (HQS)

        :param y: Degraded image.
        :param physics: Physics instance modeling the degradation.
        :param init: Initialization of the algorithm. If None, the algorithm starts from y.
        '''
        if init is None:
            x = y
        else:
            x = init
        for it in range(self.max_iter):
            x_prev = x
            if not self.g_first : 
                z = self.data_fidelity.prox(x, y, physics, self.lamb*self.stepsize[it])
                x = self.prox_g(z, it)
            else :
                z = self.prox_g(z, it)
                x = self.data_fidelity.prox(z, y, physics, self.lamb*self.stepsize[it])
            if not self.unroll and check_conv(x_prev,x,it, self.crit_conv, self.verbose) :
                break
        return x 

    def PGD(self, y, physics, init=None) : 
        '''
        Proximity Gradient Descent (PGD)

        :param y: Degraded image.
        :param physics: Physics instance modeling the degradation.
        :param init: Initialization of the algorithm. If None, the algorithm starts from y.
        '''
        if self.prox_g is None and self.grad_g is not None:
            self.g_first = True 
        if init is None:
            x = y
        else:
            x = init
        for it in range(self.max_iter):
            x_prev = x
            if not self.g_first : # prox on g and grad on f
                z = x - self.stepsize[it]*self.lamb*self.data_fidelity.grad(x, y, physics)
                x = self.prox_g(z, it)
            else :  # prox on f and grad on g
                z = x - self.stepsize[it]*self.grad_g(x,it)
                x = self.data_fidelity.prox(z, y, physics, self.lamb*self.stepsize[it])
            if not self.unroll and check_conv(x_prev,x,it, self.crit_conv, self.verbose) :
                break
        return x 

    def DRS(self, y, physics, init=None):
        '''
        Douglas-Rachford Splitting (DRS)

        :param y: Degraded image.
        :param physics: Physics instance modeling the degradation.
        :param init: Initialization of the algorithm. If None, the algorithm starts from y.
        '''
        if init is None:
            x = y
        else:
            x = init
        for it in range(self.max_iter):
            x_prev = x
            if not self.g_first :
                z = self.data_fidelity.prox(x, y, physics, self.lamb*self.stepsize[it])
                w = self.prox_g(2*z-x_prev, it)
            else :
                z = self.prox_g(x, it)
                w = self.data_fidelity.prox(2*z-x_prev, y, physics, self.lamb*self.stepsize[it])
            x = x_prev + self.theta[it]*(w - z)
            if not self.unroll and check_conv(x_prev,x,it, self.crit_conv, self.verbose) :
                break
        return w

    def ADMM(self, y, physics, init=None):
        '''
        Alternating Direction Method of Multipliers (ADMM)

        :param y: Degraded image.
        :param physics: Physics instance modeling the degradation.
        :param init: Initialization of the algorithm. If None, the algorithm starts from y.
        '''
        if init is None:
            w = y
        else:
            w = init
        x = torch.zeros_like(w)
        for it in range(self.max_iter):
            x_prev = x
            if not self.g_first :
                z = self.data_fidelity.prox(w-x, y, physics, self.lamb*self.stepsize[it])
                w = self.prox_g(z+x_prev, it)
            else :
                z = self.prox_g(w-x, it)
                w = self.data_fidelity.prox(z+x_prev, y, physics, self.lamb*self.stepsize[it])
            x = x_prev + self.theta[it]*(z - w)
            if not self.unroll and check_conv(x_prev,x,it, self.crit_conv, self.verbose) :
                break
        return w


    def forward(self, y, physics, init=None):
        if self.algo_name == 'HQS':
            return self.HQS(y, physics, init)
        elif self.algo_name == 'GD':
            return self.GD(y, physics, init)
        elif self.algo_name == 'PGD':
            return self.PGD(y, physics, init)
        elif self.algo_name == 'DRS':
            return self.DRS(y, physics, init)
        elif self.algo_name == 'ADMM':
            return self.ADMM(y, physics, init)
        else:
            raise ValueError('Unknown algorithm name')





    

        

