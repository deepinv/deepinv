import torch
import fastmri
from fastmri.data import transforms as T
from deepinv.diffops.physics.forward import Physics

class MRI(Physics):
    r'''
    Undersampled FFT operator for MRI image reconstruction problems.
    
    dependency: 'fastmri' library
    '''
    def __init__(self, acceleration=4, device='cpu', **kwargs):
        '''
        :param acceleration: (int), downsampling factor, 1/acceleration measurements will be sampled and used.
        :param device:  (str) options = 'cpu', 'cuda=0'
        '''
        super().__init__(**kwargs)
        mask = torch.load(f'../physics/mask_{acceleration}x.pth.tar')['mask']

        self.mask = mask.to(device)
        self.mask_func = lambda shape, seed: self.mask

        self.name = 'mri'
        self.acceleration = acceleration
        self.compress_ratio = 1 / acceleration


    def forward(self, x):
        # fft
        y = fastmri.fft2c(x.permute(0, 2, 3, 1))
        # add noise
        y = self.noise(y)
        # sampling by masking
        y, _ = T.apply_mask(y, self.mask_func)
        return y.permute(0,3,1,2)

    def apply_mask(self, y):
        # sampling by masking
        y, _ = T.apply_mask(y, self.mask_func)
        return y

    def A(self, x):
        # fft
        y = fastmri.fft2c(x.permute(0, 2, 3, 1))
        # sampling by masking
        y, _ = T.apply_mask(y, self.mask_func)
        return y.permute(0, 3, 1, 2)

    def A_dagger(self, y):
        # ifft 
        x = fastmri.ifft2c(y.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2)
    
    def A_adjoint(self, y):
        # ifft
        return self.A_dagger(y)
