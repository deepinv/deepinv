import torch
import torch.nn as nn
from deepinv.optim.optim_iterator import ProxOptim

class RED(ProxOptim):
    '''
    Regulatization by Denoiser (RED) algorithms for Image Restoration. Consists in replacing Id-grad_g with a denoiser.

    :param denoiser: Dennoiser model
    :param sigma_denoiser: Denoiser noise standart deviation.
    '''
    def __init__(self, denoiser, sigma_denoiser = 0.05, **kwargs):
        super().__init__(**kwargs, grad_g=lambda x,it:x)

        assert self.algo_name in ['GD','PGD'], 'RED only works with GD or PGD'

        self.denoiser = denoiser

        if self.unroll and not self.weight_tied:
            self.denoiser = torch.nn.ModuleList([denoiser for _ in range(self.max_iter)])
        else:
            self.denoiser = denoiser

        if isinstance(sigma_denoiser, float):
            sigma_denoiser = [sigma_denoiser] * self.max_iter
        elif isinstance(sigma_denoiser, list):
            assert len(sigma_denoiser) == self.max_iter
            sigma_denoiser = sigma_denoiser
        else:
            raise ValueError('sigma_denoiser must be either int/float or a list of length max_iter') 
        if self.unroll : 
             self.register_parameter(name='sigma_denoiser',
                            param=torch.nn.Parameter(torch.tensor(sigma_denoiser, device=self.device),
                            requires_grad=True))
        else:
            self.sigma_denoiser = sigma_denoiser

        self.grad_g = lambda x,it : x-self.denoiser[it](x, self.sigma_denoiser[it]) if self.unroll and not self.weight_tied else x-self.denoiser(x, self.sigma_denoiser[it])