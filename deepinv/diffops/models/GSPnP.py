import numpy as np
import torch
import torch.nn as nn

from .denoiser import register

class StudentGrad(nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.model = denoiser
    def forward(self, x, sigma):
        return self.model(x, sigma)

class GSPnP(nn.Module):
    '''
    Gradient Step module to use a denoiser architecture as a Gradient Step Denoiser.

    :param denoiser: (nn.Module) Denoiser module
    '''
    def __init__(self, denoiser, train = False):
        super().__init__()
        self.student_grad = StudentGrad(denoiser)
        self.train = train

    def potential(self, x, sigma):
        N = self.student_grad(x, sigma)
        return 0.5*torch.norm(x-N)**2

    def potential_grad(self, x, sigma):
        '''
        Calculate Dg(x) the gradient of the regularizer g at input x
        :param x: torch.tensor Input image
        :param sigma: Denoiser level (std)
        :return: Dg(x), DRUNet output N(x)
        '''
        torch.set_grad_enabled(True)
        x = x.float()
        x = x.requires_grad_()
        N = self.student_grad(x, sigma)
        JN = torch.autograd.grad(N, x, grad_outputs=x - N, create_graph=True, only_inputs=True)[0]
        if not self.train:
            torch.set_grad_enabled(False)
        Dg = x - N - JN
        return Dg

    def forward(self, x, sigma):
        '''
        Denoising with Gradient Step Denoiser
        :param x:  torch.tensor input image
        :param sigma: Denoiser level (std)
        :return: Denoised image x_hat, Dg(x) gradient of the regularizer g at x
        '''
        Dg = self.potential_grad(x, sigma)
        x_hat = x - Dg
        return x_hat

@register('gsdrunet')
def GSDRUNet(in_channels=4, out_channels=3, nb=2, nc=[64, 128, 256, 512], act_mode='E', pretrain=False, ckpt_path=None, train=False, device=torch.device('cpu')):
    '''
    Gradient Step DRUNet
    :param in_channels: (int) Number of input channels
    :param out_channels: (int) Number of output channels
    :param nb: (int) Number of blocks in the DRUNet
    :param nc: (list) Number of channels in the DRUNet
    '''
    from deepinv.diffops.models.drunet import DRUNet
    denoiser = DRUNet(in_channels=in_channels, out_channels=out_channels, nb=nb, nc=nc, act_mode=act_mode, pretrain=False, train=train, device=device)
    model = GSPnP(denoiser, train=train)
    if pretrain and ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path), strict=False)
    return model