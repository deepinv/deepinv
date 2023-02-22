import torch
import torch.nn as nn

class PnP(nn.Module):
    '''
    Plug-and-Play algorithms for Image Restoration. Consists in replacing prox_g with a denoiser.

    :param denoiser: Dennoiser model
    :param sigma_denoiser: Denoiser noise standart deviation.
    '''
    def __init__(self, denoiser, sigma_denoiser=0.05, max_iter=50, unroll=False, weight_tied=False, device = 'cpu'):
        super().__init__()

        self.denoiser = denoiser
        self.unroll = unroll
        self.weight_tied = weight_tied
        self.max_iter=max_iter
        self.device=device

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
    
    def prox_g(self, x,it) : 
        return self.denoiser[it](x, self.sigma_denoiser[it]) if self.unroll and not self.weight_tied else self.denoiser(x, self.sigma_denoiser[it])
