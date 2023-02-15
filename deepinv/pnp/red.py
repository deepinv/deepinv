import torch
import torch.nn as nn
from deepinv.optim.optim_base import ProxOptim

class RED(ProxOptim):
    '''
    '''
    def __init__(self, denoiser, sigma_denoiser = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.denoiser = denoiser
        if not self.unroll : 
            if isinstance(sigma_denoiser, float):
                self.sigma_denoiser = [sigma_denoiser] * self.max_iter
            elif isinstance(sigma_denoiser, list):
                assert len(sigma_denoiser) == self.max_iter
                self.sigma_denoiser = sigma_denoiser
            else:
                raise ValueError('sigma_denoiser must be either int/float or a list of length max_iter') 
        else : 
            assert isinstance(sigma_denoiser, float) # the initial parameter is uniform across layer int in that case
            self.register_parameter(name='sigma_denoiser',
                                param=torch.nn.Parameter(torch.tensor(sigma_denoiser, device=device),
                                requires_grad=True))

        self.grad_g = lambda x,it : x-denoiser(x, self.sigma_denoiser[it])