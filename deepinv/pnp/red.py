import torch
import torch.nn as nn
from deepinv.optim.optim_base import optim

class RED(optim):
    '''
    Regularization-by-Denoising (RED) algorithms for image restoration. 
    Consists in replacing the gradient of the regularization by the residual of denoiser.

    :param denoiser: denoising operator 
    :param sigma: Noise level of the denoiser. List or int. If list, the length of the list must be equal to max_iter.
    '''
    def __init__(self, denoiser, sigma = 1., **kwargs):
        super().__init__(**kwargs)
        
        assert self.algo_name in ('PD','PGD'), 'Algo must be gradient-based'
        if not self.unroll : 
            if isinstance(sigma, int):
                self.sigmas = [sigma] * max_iter
            elif isinstance(sigma, list):
                assert len(sigma) == max_iter
                self.sigmas = sigma
            else:
                raise ValueError('sigma must be either an int or a list of length max_iter') 
        else : 
            assert isinstance(sigma, int) # the initial parameter is uniform across layer int in that case
            self.register_parameter(name='sigmas',
                                param=torch.nn.Parameter(torch.tensor(sigma, device=device),
                                requires_grad=True))

    def GD(self, y) : 
        x = y
        for it in range(self.max_iter):
            x_prev = x
            x = x - self.stepsizes[it]*((x-self.denoiser(x, self.sigmas[it])) + self.physics.grad(x))
            if self.check_conv(x_prev,x) :
                break
        return x 


    def PGD(self, y) : 
        x = y
        for it in range(self.max_iter):
            x_prev = x
            z = self.physics.prox(x, self.stepsizes[it])
            x = z - self.stepsizes[it]*(z-self.denoiser(z, self.sigmas[it]))
            if self.check_conv(x_prev,x) :
                break
        return x 

    def forward(self, y) : 
        if self.algo_name == 'GD' : 
            return self.GD(y)
        elif self.algo_name == 'PGD' : 
            return self.PGD(y)
        else : 
            raise notImplementedError
