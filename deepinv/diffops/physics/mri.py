import torch
import fastmri
from fastmri.data import transforms as T
from deepinv.diffops.physics.forward import Physics

class MRI(Physics):
    r'''
    Undersampled FFT operator for MRI image reconstruction problems.
    
    dependency: 'fastmri' library
    '''
    def __init__(self, acceleration=4, mask=None, device='cpu', **kwargs):
        '''

        :param acceleration: (int), downsampling factor, 1/acceleration measurements will be sampled and used.
        :param mask: (tensor), (0,1) size=[1,img_width,1]
        :param device: (str) options = 'cpu', 'cuda=0'
        :param kwargs:
        '''

        super().__init__(**kwargs)

        # print('mask.shape=',mask.shape)

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



if __name__ == '__main__':
    # deepinv test
    from deepinv.tests.test_physics import test_operators_norm, test_operators_adjointness, test_pseudo_inverse, device

    test_operators_norm('MRI', (2, 320, 320), 'cuda:0') #pass
    test_pseudo_inverse('MRI', (2, 320, 320), 'cuda:0') #pass
    test_operators_adjointness('MRI', (2,320,320), 'cuda:0')#failed, error=tensor(30.6791, device='cuda:0')

    # dump test
    x = torch.randn(3,2,320,320).to(device)
    acceleration=2
    # mask = torch.load(f'mask_{acceleration}x.pth.tar')['mask']  # torch.Size([1, 320, 1])
    mask = (torch.rand(1,320,1)>1/2).type(torch.int)

    mri = MRI(acceleration=acceleration,
              mask=mask,
              device=device)
    y = mri.A(x)
    x_dagger = mri.A_dagger(y)
    x_adjoint = mri.A_adjoint(y)
    print(x.shape, y.shape, x_adjoint.shape, x_dagger.shape)


