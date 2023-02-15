import torch
import torch.nn as nn

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
    :param grad_g: Gradient of the regularizing potential. x,it -> grad_g(x)
    :param max_iter: Number of iterations.
    :param step_size: Step size of the algorithm. List or int. If list, the length of the list must be equal to max_iter.
    :param theta: Relacation parameter of the ADMM/DRS/PD algorithms.
    :param g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param crit_conv: Mimimum relative change in the solution to stop the algorithm.
    :param unroll: If True, the algorithm is unrolled in time.
    :param verbose: prints progress of the algorithm.
    '''

    def __init__(self, algo_name='PGD', data_fidelity='L2', lamb=1., device='cpu', g = None, prox_g = None, grad_g = None, max_iter=10, stepsize = 1., theta = 1., g_first = False, crit_conv=None, unroll = False, verbose=False):
        super().__init__()

        self.algo_name = algo_name
        self.data_fidelity = data_fidelity
        self.lamb = lamb
        self.g = g
        self.prox_g = prox_g
        self.grad_g = grad_g
        self.g_first = g_first
        self.unroll = unroll
        self.max_iter = max_iter
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.device = device

        if not unroll : 
            if isinstance(stepsize, float):
                self.stepsize = [stepsize] * max_iter
            elif isinstance(stepsize, list):
                assert len(stepsize) == max_iter
                self.stepsize = stepsize
            else:
                raise ValueError('stepsize must be either int/float or a list of length max_iter') 
        else : 
            assert isinstance(stepsize, float) # the initial parameter is uniform across layer int in that case
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


    def check_conv(self,x_prev,x,it):
        crit_cur = (x_prev-x).norm() / (x.norm()+1e-03)
        if self.verbose:
            print(it, 'crit = ', crit_cur , '\r')
        if self.crit_conv is not None and crit_cur < self.crit_conv:
            return True
        else :
            return False

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
            if not self.unroll and self.check_conv(x_prev,x,it) :
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
            if not self.unroll and self.check_conv(x_prev,x,it) :
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
            if not self.unroll and self.check_conv(x_prev,x,it) :
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
            if not self.unroll and self.check_conv(x_prev,x,it) :
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
            if not self.unroll and self.check_conv(x_prev,x,it) :
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





    

        

