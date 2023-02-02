import torch
import fastmri
from fastmri.data import transforms as T
from deepinv.diffops.physics.forward import Forward

class MRI(Forward):
    '''
    TODO
    '''
    def __init__(self, acceleration=4, device='cpu'):
        '''

        TODO

        :param acceleration: 
        :param device: 
        '''
        super().__init__()
        mask = torch.load(f'../physics/mask_{acceleration}x.pth.tar')['mask']

        self.mask = mask.to(device)
        self.mask_func = lambda shape, seed: self.mask

        self.name = 'mri'
        self.acceleration = acceleration
        self.compress_ratio = 1 / acceleration


    def forward(self, x):
        y = fastmri.fft2c(x.permute(0, 2, 3, 1))
        y = self.noise(y)
        y, _ = T.apply_mask(y, self.mask_func)
        return y.permute(0,3,1,2)

    def apply_mask(self, y):
        y, _ = T.apply_mask(y, self.mask_func)
        return y

    def A(self, x, add_noise=False):
        y = fastmri.fft2c(x.permute(0, 2, 3, 1))
        # no noise
        y, _ = T.apply_mask(y, self.mask_func)
        return y.permute(0, 3, 1, 2)

    def A_dagger(self, y):
        x = fastmri.ifft2c(y.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2)