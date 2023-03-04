import torch
import torch.nn as nn

class PnP(nn.Module):
    '''
    Plug-and-Play (PnP) / Regularization bu Denoising (RED) algorithms for Image Restoration. Consists in replacing prox_g or grad_g with a denoiser.
    '''
    def __init__(self, denoiser, init=None, stepsize=1., sigma_denoiser=0.05, max_iter=50,
                 device = torch.device('cpu')):
        super().__init__()

        self.denoiser = denoiser
        self.max_iter = max_iter
        self.device = device
        self.init = init
            
        if isinstance(sigma_denoiser, float):
            self.sigma_denoiser = [sigma_denoiser] * self.max_iter
        elif isinstance(sigma_denoiser, list):
            assert len(sigma_denoiser) == self.max_iter
            self.sigma_denoiser = sigma_denoiser
        else:
            raise ValueError('sigma_denoiser must be either float or a list of length max_iter')

        if isinstance(stepsize, float):
            self.stepsize = [stepsize]*max_iter  # Should be a list
        elif isinstance(stepsize, list):
            assert len(stepsize) == self.max_iter
            self.stepsize = stepsize
        else:
            raise ValueError('stepsize must be either float or a list of length max_iter')

    def prox_g(self, x, sigma, it):
        if isinstance(self.denoiser, list) or isinstance(self.denoiser, nn.ModuleList):
            out = self.denoiser[it](x, sigma)
        else:
            out = self.denoiser(x, sigma)
        return out

    def grad_g(self, x, sigma, it):
        if isinstance(self.denoiser, list) or isinstance(self.denoiser, nn.ModuleList):
            out = x - self.denoiser[it](x, sigma)
        else:
            out = x - self.denoiser(x, sigma)
        return out