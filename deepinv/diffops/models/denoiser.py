import torch
import torch.nn as nn
import numpy as np

class Denoiser(nn.Module):
    def __init__(self, denoiser_name, device, n_channels=3, pretrain=False, ckpt_path=None, train=False):
        '''
        Image Denoiser parent class
        :param denoiser_name: str, name of the denoiser
        :param device: torch.device
        :param n_channels: int, number of channels of the input image
        :param pretrain: bool, whether to load pretrained weights
        :param ckpt_path: str, path to the pretrained weights
        :param train: bool, whether to train the denoiser or not
        '''
        super(Denoiser, self).__init__()
        self.device = device
        self.denoiser_name = denoiser_name
        
        if 'drunet' in self.denoiser_name:
            from deepinv.diffops.models.drunet import DRUNet
            if 'tiny' in self.denoiser_name:
                self.model = DRUNet(in_channels=n_channels+1, out_channels=n_channels, nb = 2, nc=[16, 32, 64, 64])
            else :
                self.model = DRUNet(in_channels=n_channels+1, out_channels=n_channels, nb = 4, nc=[64, 128, 256, 512])
            deep = True
        elif 'GSDRUNet' in self.denoiser_name:
            from deepinv.diffops.models.GSPnP import GSDRUNet
            self.model = GSDRUNet(in_channels=n_channels+1, out_channels=n_channels, nb = 2, nc=[64, 128, 256, 512], act_mode='E')
        elif self.denoiser_name == 'TGV':
            from deepinv.diffops.models.tgv import TGV
            self.model = TGV(n_it_max=100, verbose=False)
            deep = False
        elif self.denoiser_name == 'BM3D':
            import bm3d
            from utils import tensor2array, array2tensor
            self.model = lambda x,sigma : torch.cat([array2tensor(bm3d.bm3d(tensor2array(xi), sigma)) for xi in x])
            deep = False
        else: 
            raise Exception("The denoiser chosen doesn't exist")

        if isinstance(self.model, nn.Module):
            if pretrain and ckpt_path is not None:
                self.model.load_state_dict(torch.load(ckpt_path), strict=False)
            if not train:
                self.model.eval()
                for _, v in self.model.named_parameters():
                    v.requires_grad = False
            self.model = self.model.to(device)
        
    def forward(self, x, sigma):
        return self.model(x, sigma)