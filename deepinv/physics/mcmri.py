import torch
from torch import nn
from deepinv.physics.forward import LinearPhysics

from deepinv.physics import adjoint_function




class MultiCoilMRI(LinearPhysics):
    r"""
    Multi-coil MRI operator.

    The linear operator is defined as:
    .. math::

        y_c = \text{diag}(p) F \text{diag}(s_c) x

        for c=1,\dots,C coils, where y_c are the measurements from the cth coil, \text{diag}(p) is the acceleration mask, F is the Fourier transform and \diag(s_c) is the cth coil sensitivity.
    
    :param torch.Tensor mask: binary sampling mask which should have shape [B,1,H,W].
    :param torch.Tensor coil_maps: complex valued coil sensitvity maps which should have shape [B,C,H,W].
    :param device: specify which device you want to use (i.e, cpu or gpu).
    """

    def __init__(
        self,
        mask,
        coil_maps,
        device=torch.device("cpu"),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mask = mask.to(device)
        self.coil_maps = coil_maps.to(device)

    def A(self, x, **kwargs):
        r"""  
        Applies linear operator.  
        
        :param torch.Tensor x: image with shape [B,H,W,2].  
        :returns: (torch.Tensor) multi-coil kspace measurements with shape [B,C,H,W,2].  
        """ 
        x_cplx = torch.view_as_complex(x)
        coil_imgs = self.coil_maps*x_cplx[:,None,...]
        coil_ksp = fft(coil_imgs)
        output = self.mask*coil_ksp
        return torch.view_as_real(output)

    def A_adjoint(self, y, **kwargs):
        r"""  
        Applies adjoint linear operator.  
        
        :param torch.Tensor y: multi-coil kspace measurements with shape [B,C,H,W,2].  
        :returns: (torch.Tensor) image with shape [B,H,W,2]  
        """ 
        sampled_ksp = self.mask* torch.view_as_complex(y)
        coil_imgs = ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(self.coil_maps)*coil_imgs,dim=1) 
        img_out_2ch = torch.view_as_real(img_out)
        return img_out_2ch 
    

# Centered, orthogonal ifft 
def ifft(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x

# Centered, orthogonal fft
def fft(x):
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    return x