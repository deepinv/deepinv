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
        super().__init__(**kwargs)

        assert self.algo_name in ['PGD','ADMM','DRS'], 'PnP only works with PGD, ADMM or DRS'

        self.denoiser = denoiser
        if isinstance(sigma_denoiser, float):
            self.sigma_denoiser = [sigma_denoiser] * self.max_iter
        elif isinstance(sigma_denoiser, list):
            assert len(sigma_denoiser) == self.max_iter
            self.sigma_denoiser = sigma_denoiser
        else:
            raise ValueError('sigma_denoiser must be either int/float or a list of length max_iter') 

        self.prox_g = lambda x,it : self.denoiser(x, self.sigma_denoiser[it])


class UnrolledPnP(ProxOptim):
    '''
    Unrolled Plug-and-Play algorithms for Image Restoration. 

    :param denoiser: Dennoiser model
    :param sigma_denoiser: Denoiser noise standart deviation.
    '''
    def __init__(self, backbone_net,  weight_tied=False, sigma_denoiser=0.05, **kwargs):
        super().__init__(**kwargs)

        self.weight_tied = weight_tied
        if self.weight_tied:
            self.blocks = torch.nn.ModuleList([backbone_net])
        else:
            self.blocks = torch.nn.ModuleList([backbone_net for _ in range(self.max_iter)])
        self.blocks.to(self.device)
        if isinstance(sigma_denoiser, float):
            sigma_denoiser = [sigma_denoiser] * self.max_iter
        elif isinstance(sigma_denoiser, list):
            assert len(sigma_denoiser) == self.max_iter
        else:
            raise ValueError('sigma_denoiser must be either int/float or a list of length max_iter') 
        self.register_parameter(name='sigma_denoiser',
                            param=torch.nn.Parameter(torch.tensor(sigma_denoiser, device=self.device),
                            requires_grad=True))

        self.prox_g = lambda x,it : self.blocks[it](x, self.sigma_denoiser[it]) if self.weight_tied else self.blocks[0](x, self.sigma_denoiser[it])
