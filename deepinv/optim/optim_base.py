import torch
import torch.nn as nn

class optim(nn.Module):
    '''
    Optimization algorithms for minimizing the sum of two functions f + g where f is a data-fidelity term that will me modeled by an instance of physics
    and g is a regularizer either explicit or implicitly given by either its prox or its gradient. 

    :param algo_name: Name of the optimization algorithm.
    :param physics: Physics instance modeling the data-fidelity term.
    :param g: Regularizing potential, None for an implicit regularizer.
    :param max_iter: Number of iterations.
    :param step_size: Step size of the algorithm. List or int. If list, the length of the list must be equal to max_iter.
    :param crit_conv: Mimimum relative change in the solution to stop the algorithm.
    :param unroll: If True, the algorithm is unrolled in time.
    :param verbose: prints progress of the algorithm.
    '''

    def __init__(self, algo_name, physics, device, g = None, max_iter=10, stepsize = 1, crit_conv=None, unroll = False, verbose=False):
        super(optim, self).__init__()

        assert algo_name in ('GD', 'HQS', 'PGD', 'ADMM', 'DRS', 'PD'), 'Optimization algorithm not implemented'
        self.algo_name = algo_name
        self.g = g
        self.physics = physics
        self.unroll = unroll
        if not unroll : 
            if isinstance(stepsize, int):
                self.stepsizes = [stepsize] * max_iter
            elif isinstance(stepsize, list):
                assert len(stepsize) == max_iter
                self.stepsizes = stepsize
            else:
                raise ValueError('stepsize must be either an int or a list of length max_iter') 
        else : 
            assert isinstance(stepsize, int) # the initial parameter is uniform across layer int in that case
            self.register_parameter(name='step_size',
                                param=torch.nn.Parameter(torch.tensor(stepsize, device=device),
                                requires_grad=True))
        self.max_iter = max_iter
        self.crit_conv = crit_conv

    def check_conv(x_prev,x):
        crit_cur = (x_prev-x).norm() / (x.norm()+1e-03)
        if self.verbose:
            print(it, 'crit = ', crit_cur , '\r')
        if self.crit_conv is not None and crit_cur < self.crit_conv:
            return True
        else :
            return False




    

        

