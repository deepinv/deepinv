import torch
import torch.nn as nn
from deepinv.optim.optim_base import optim


class PnP(optim):
    '''
    Plug-and-Play (PnP) algorithms for image restoration. 
    Consists in replacing the proximal operator of the regularization by a denoiser.

    :param denoiser: denoising operator 
    :param sigma: Noise level of the denoiser. List or int. If list, the length of the list must be equal to max_iter.
    '''
    def __init__(self, denoiser, sigma = 1/255., theta = 1., **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

        assert self.algo_name in ('HQS', 'PGD', 'ADMM', 'DRS', 'PD'), 'Algo must be proximal'
        if not self.unroll : 
            if isinstance(sigma, float) or isinstance(sigma, int):
                self.sigmas = [sigma] * self.max_iter
            elif isinstance(sigma, list):
                assert len(sigma) == self.max_iter
                self.sigmas = sigma
            else:
                raise ValueError('sigma must be either int/float,  or a list of length max_iter') 
        else : 
            assert isinstance(sigma, int) # the initial parameter is uniform across layer int in that case
            self.register_parameter(name='sigmas',
                                param=torch.nn.Parameter(torch.tensor(sigma, device=device),
                                requires_grad=True))

        
    def HQS(self, y, physics, init=None) : 
        '''
        Plug-and-Play Half Quadratric Splitting (HQS) algorithm for image restoration.

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
            z = self.data_fidelity.prox(x, y, physics, self.stepsizes[it])
            x = self.denoiser(z, self.sigmas[it])
            if not self.unroll and self.check_conv(x_prev,x) :
                break
        return x 

    def PGD(self, y, physics, init=None) : 
        if init is None:
            x = y
        else:
            x = init
        for it in range(self.max_iter):
            x_prev = x
            z = x - self.stepsizes[it]*self.data_fidelity.grad(x, y, physics)
            x = self.denoiser(z, self.sigmas[it])
            if not self.unroll and self.check_conv(x_prev,x) :
                break
        return x 

    def DRS(self, y, physics, init=None) :
        # TO DO 
        pass


    def ADMM(self, y, physics, init=None):
        # TO DO 
        pass

    def PD(self, y, physics, init=None):
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








