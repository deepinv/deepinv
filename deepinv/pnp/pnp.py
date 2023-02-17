import torch
import torch.nn as nn
from deepinv.optim.optim_base import ProxOptim

class PnP(ProxOptim):
    '''
    Plug-and-Play algorithms for Image Restoration. Consists in replacing prox_g with a denoiser.

    :param denoiser: Dennoiser model
    :param sigma_denoiser: Denoiser noise standart deviation.
    '''
    def __init__(self, denoiser, sigma_denoiser=0.05, **kwargs):
        super().__init__(**kwargs, prox_g = lambda x,it:x)

        assert self.algo_name in ['HQS','PGD','ADMM','DRS'], 'PnP only works with HQS, PGD, ADMM or DRS'

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

        self.prox_g = lambda x,it : self.denoiser[it](x, self.sigma_denoiser[it]) if self.unroll and not self.weight_tied else self.denoiser(x, self.sigma_denoiser[it])
