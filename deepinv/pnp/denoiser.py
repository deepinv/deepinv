import torch
import torch.nn as nn
import bm3d

class Denoiser(nn.Module):
    def __init__(self, denoiser_name):
        '''
        '''
        super(Denoiser, self).__init__()
        self.denoiser_name = denoiser_name
        
    def forward(self, x, sigma):
        if self.denoiser_name == 'BM3D':
            return bm3d.bm3d(x, sigma)
        else:
            raise NotImplementedError