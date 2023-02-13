import torch
import torch.nn as nn
from deepinv.optim.optim_base import optim

class PnP(optim):
    '''
    Plug-and-Play (PnP) algorithms for image restoration. 
    Consists in replacing the proximal operator of the regularization by a denoiser.

    :param denoiser: denoising operator 
    :param sigma: Noise level of the denoiser. List or int. If list, the length of the list must be equal to max_iter.
    :param theta: Additional parameter for ADMM and DRS.
    '''
    def __init__(self, denoiser, sigma = 1., theta = 1., **kwargs):
        super().__init__(**kwargs)

        assert self.algo_name in ('HQS', 'PGD', 'ADMM', 'DRS', 'PD'), 'Algo must be proximal'
        if not self.unroll : 
            if isinstance(sigma, float) or isinstance(sigma, int):
                self.sigmas = [sigma] * self.max_iter
            elif isinstance(sigma, list):
                assert len(sigma) == self.max_iter
                self.sigmas = sigma
            else:
                raise ValueError('sigma must be either an int or a list of length max_iter') 
        else : 
            assert isinstance(sigma, int) # the initial parameter is uniform across layer int in that case
            self.register_parameter(name='sigmas',
                                param=torch.nn.Parameter(torch.tensor(sigma, device=device),
                                requires_grad=True))

        if isinstance(theta, float) or isinstance(theta, int):
            self.thetas = [theta] * self.max_iter
        elif isinstance(theta, list):
            assert len(theta) == self.max_iter
            self.thetas = theta
        else:
            raise ValueError('theta must be either an int or a list of length max_iter') 

        
    def HQS(self, y) : 
        x = y
        for it in range(self.max_iter):
            x_prev = x
            z = self.physics.prox(x, self.stepsizes[it])
            x = self.denoiser(z, self.sigmas[it])
            if self.check_conv(x_prev,x) :
                break
        return x 

    def PGD(self, y) : 
        x = y
        for it in range(self.max_iter):
            x_prev = x
            z = x - self.stepsizes[it]*self.physics.grad(x)
            x = self.denoiser(z, self.sigmas[it])
            if self.check_conv(x_prev,x) :
                break
        return x 

    def DRS(self, y) :
        x = y
        for it in range(self.max_iter):
            x_prev = x
            z = 2*self.physics.prox(x, self.stepsizes[it]) - x
            y = 2*self.denoiser(z, self.sigmas[it]) - z
            x = self.thetas[it]*y + (1-self.theta[it])*x_prev
            if self.check_conv(x_prev,x) :
                break
        return x

    def ADMM(self, y):
        # TO DO 
        pass

    def PD(self, y):
        # TO DO 
        pass

    def forward(self, y):
        if self.algo_name == 'HQS':
            return self.HQS(y)
        elif self.algo_name == 'PGD':
            return self.PGD(y)
        elif self.algo_name == 'DRS':
            return self.DRS(y)
        elif self.algo_name == 'ADMM':
            return self.ADMM(y)
        elif self.algo_name == 'PD':
            return self.PD(y)
        else:
            raise notImplementedError








