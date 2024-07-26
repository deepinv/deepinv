import torch
from torch import nn
from deepinv.physics.forward import LinearPhysics

from deepinv.physics import adjoint_function




class mcMRI(LinearPhysics):
    r"""
    Multi-coil MRI operator.

    Here the linear operator is defined as:
    .. math::

        y = PSFx

    where :math:`P` is the subsampling mask, :math:`S` are the coil sensitivity maps, :math:`F` is the Fourier transform.

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
        x_cplx = torch.view_as_complex(x) # input is real [B,H,W,2]
        coil_imgs = self.coil_maps*x_cplx[:,None,...]
        coil_ksp = fft(coil_imgs)
        output = self.mask*coil_ksp
        return torch.view_as_real(output) # output is real [B,C,H,W,2]

    def A_adjoint(self, y, **kwargs):
        sampled_ksp = self.mask* torch.view_as_complex(y) # assumes y is real [B,C,H,W,2]
        coil_imgs = ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(self.coil_maps)*coil_imgs,dim=1) 
        img_out_2ch = torch.view_as_real(img_out) # output is real [B,H,W,2]
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