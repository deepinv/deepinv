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
    
    :param torch.Tensor mask: binary sampling mask which should have shape [B,1,N,H,W].
    :param torch.Tensor coil_maps: complex valued coil sensitvity maps which should have shape [B,C,N,H,W].
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
        
        :param torch.Tensor x: image with shape [B,2,N,H,W].  
        :returns: (torch.Tensor) multi-coil kspace measurements with shape [B,2,C,N,H,W].
        """
        x_cplx = torch.view_as_complex(x.permute(0,-3,-2,-1,1))[:,None,...] # outputs [B,N,H,W]
        coil_imgs = self.coil_maps*x_cplx # outputs [B,C,N,H,W]
        coil_ksp = fft(coil_imgs)
        output = self.mask*coil_ksp # outputs [B,C,N,H,W]
        return torch.view_as_real(output).permute(0,-1,-5,-4,-3,-2) # outputs [B,2,C,N,H,W]

    def A_adjoint(self, y, **kwargs):
        r"""  
        Applies adjoint linear operator.  
        
        :param torch.Tensor y: multi-coil kspace measurements with shape [B,2,C,N,H,W].
        :returns: (torch.Tensor) image with shape [B,2,N,H,W]
        """ 
        sampled_ksp = self.mask* torch.view_as_complex(y.permute(0,-4,-3,-2,-1,1)) # outputs [B,C,N,H,W]
        coil_imgs = ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(self.coil_maps)*coil_imgs,dim=1)
        img_out_2ch = torch.view_as_real(img_out).permute(0,-1,-4,-3,-2) # outputs [B,2,N,H,W]
        return img_out_2ch
    

# Centered, orthogonal ifft 
def ifft(x):
    x = torch.fft.ifftshift(x, dim=(-3, -2, -1))
    x = torch.fft.ifftn(x, dim=(-3, -2, -1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-3, -2, -1))
    return x

# Centered, orthogonal fft
def fft(x):
    x = torch.fft.fftshift(x, dim=(-3, -2, -1))
    x = torch.fft.fftn(x, dim=(-3, -2, -1), norm='ortho')
    x = torch.fft.ifftshift(x, dim=(-3, -2, -1))
    return x