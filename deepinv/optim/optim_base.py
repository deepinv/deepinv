import torch
import torch.nn as nn

class optim(nn.Module):
    '''
    Optimization algorithms for minimizing the sum of two functions f + g where f is a data-fidelity term that will me modeled by an instance of physics
    and g is a regularizer either explicit or implicitly given by either its prox or its gradient. 
    By default, the algorithms starts with a step on f and finishes with step on g. 

    :param algo_name: Name of the optimization algorithm.
    :param data_fidelity: data_fidelity instance modeling the data-fidelity term.   
    :param g: Regularizing potential. 
    :param prox_g: Proximal operator of the regularizing potential.
    :param grad_g: Gradient of the regularizing potential.
    :param max_iter: Number of iterations.
    :param step_size: Step size of the algorithm. List or int. If list, the length of the list must be equal to max_iter.
    :param theta: Relacation parameter of the ADMM/DRS/PD algorithms.
    :param g_first: If True, the algorithm starts with a step on g and finishes with a step on f.
    :param crit_conv: Mimimum relative change in the solution to stop the algorithm.
    :param unroll: If True, the algorithm is unrolled in time.
    :param verbose: prints progress of the algorithm.
    '''

    def __init__(self, algo_name, data_fidelity, device, g = None, prox_g = None, grad_g = None, max_iter=10, stepsize = 1., theta = 1., g_first = False, crit_conv=None, unroll = False, verbose=False):
        super(optim, self).__init__()

        assert algo_name in ('GD', 'HQS', 'PGD', 'ADMM', 'DRS', 'PD'), 'Optimization algorithm not implemented'
        self.algo_name = algo_name
        self.data_fidelity = data_fidelity
        self.g = g
        self.prox_g = prox_g
        self.grad_g = grad_g
        self.g_first = g_first
        self.unroll = unroll
        if not unroll : 
            if isinstance(stepsize, float):
                self.stepsizes = [stepsize] * max_iter
            elif isinstance(stepsize, list):
                assert len(stepsize) == max_iter
                self.stepsizes = stepsize
            else:
                raise ValueError('stepsize must be either int/float or a list of length max_iter') 
        else : 
            assert isinstance(stepsize, float) # the initial parameter is uniform across layer int in that case
            self.register_parameter(name='step_size',
                                param=torch.nn.Parameter(torch.tensor(stepsize, device=device),
                                requires_grad=True))
        if isinstance(stepsize, float):
            self.thetas = [theta] * max_iter
        elif isinstance(theta, list):
            assert len(theta) == max_iter
            self.thetas = theta
        else:
            raise ValueError('stepsize must be either int/float or a list of length max_iter') 

        self.max_iter = max_iter
        self.crit_conv = crit_conv
        self.verbose = verbose

    def check_conv(self,x_prev,x):
        crit_cur = (x_prev-x).norm() / (x.norm()+1e-03)
        if self.verbose:
            print(it, 'crit = ', crit_cur , '\r')
        if self.crit_conv is not None and crit_cur < self.crit_conv:
            return True
        else :
            return False


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
            if not g_first : 
                z = self.data_fidelity.prox(x, y, physics, self.stepsizes[it])
                x = self.prox_g(z, self.stepsizes[it])
            else :
                z = self.prox_g(x, self.stepsizes[it])
                x = self.data_fidelity.prox(z, y, physics, self.stepsizes[it])
            if not self.unroll and self.check_conv(x_prev,x) :
                break
        return x 

    def PGD(self, y, physics, init=None) : 
        '''
        Proximity Gradient Descent (PGD)

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
            if not g_first : # prox on g and grad on f
                z = x - self.stepsizes[it]*self.data_fidelity.grad(x, y, physics)
                x = self.prox_g(z, self.stepsizes[it])
            else :  # prox on f and grad on g
                z = x - self.stepsizes[it]*self.grad_g(x)
                x = self.data_fidelity.prox(z, y, physics, self.stepsizes[it])
            if not self.unroll and self.check_conv(x_prev,x) :
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
            if not g_first :
                z = self.data_fidelity.prox(x, y, physics, self.stepsizes[it])
                w = self.prox_g(2*z-x_prev, self.stepsizes[it])
            else :
                z = self.prox_g(x, self.stepsizes[it])
                w = self.data_fidelity.prox(2*z-x_prev, y, physics, self.stepsizes[it])
            x = x_prev + self.theta[it]*(w - z)
            if not self.unroll and self.check_conv(x_prev,x) :
                break
        return x

    def ADMM(self, y, physics, init=None):
        '''
        Alternating Direction Method of Multipliers (ADMM)

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
            if not g_first :
                z = self.data_fidelity.prox(x, y, physics, self.stepsizes[it])
                w = self.prox_g(z+x_prev, self.stepsizes[it])
            else :
                z = self.prox_g(x, self.stepsizes[it])
                w = self.data_fidelity.prox(z+x_prev, y, physics, self.stepsizes[it])
            x = x_prev + self.theta[it]*(z - w)
            if not self.unroll and self.check_conv(x_prev,x) :
                break
        return x

    def PD(self, y, physics, init=None):
        '''
        Primal-Dual (PD)

        :param y: Degraded image.
        :param physics: Physics instance modeling the degradation.
        :param init: Initialization of the algorithm. If None, the algorithm starts from y.
        '''
        # TO DO 
        pass

    def forward(self, y, physics, init=None):
        if self.algo_name == 'HQS':
            return self.HQS(y, physics, init)
        elif self.algo_name == 'PGD':
            return self.PGD(y, physics, init)
        elif self.algo_name == 'DRS':
            return self.DRS(y, physics, init)
        elif self.algo_name == 'ADMM':
            return self.ADMM(y, physics, init)
        elif self.algo_name == 'PD':
            return self.PD(y, physics, init)
        else:
            raise notImplementedError





    

        

