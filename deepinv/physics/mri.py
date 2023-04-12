import torch
import torch.fft
from typing import List, Optional
from deepinv.physics.forward import Physics

class MRI(Physics):
    r'''
    Undersampled FFT operator for MRI image reconstruction problems.
    '''
    def __init__(self, mask=None, device='cpu', **kwargs):
        '''
        :param mask: (tensor), (0,1) size=[img_width,img_height]
        :param device: (str) options = 'cpu', 'cuda=0'
        :param kwargs:
        '''
        super().__init__(**kwargs)
        self.mask = mask.to(device)
        self.device = device


    def forward(self, x):
        '''
        :param x: (Tensor) Complex valued input data (N*2*H*W) containing at least
                  3 dimensions: dimensions -2 & -1 are spatial dimensions and dimension -3
                  has size 2. All other dimensions are assumed to be batch dimensions.
        :return: The (undersampled & noised) FFT of the x.
        '''
        y = fft2c_new(x.permute(0, 2, 3, 1)) # N2HW -> NHW2
        y = self.noise(y)
        y = apply_mask(y, self.mask)
        return y.permute(0,3,1,2)

    def A(self, x):
        '''
        :param x: (Tensor) Complex valued input data (N*2*H*W) containing at least
                  3 dimensions: dimensions -2 & -1 are spatial dimensions and dimension -3
                  has size 2. All other dimensions are assumed to be batch dimensions.
        :return: The (undersampled & clean) FFT of the x.
        '''
        y = fft2c_new(x.permute(0, 2, 3, 1))
        y = apply_mask(y, self.mask)
        return y.permute(0,3,1,2)


    def A_adjoint(self, y):
        y = apply_mask(y.permute(0, 2, 3, 1), self.mask)
        x = ifft2c_new(y)
        return x.permute(0, 3, 1, 2)


    def A_dagger(self, x):
        return self.A_adjoint(x)

# reference: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -2 & -1 are spatial dimensions and dimension -3 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -2 & -1 are spatial dimensions and dimension -3 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# Helper functions
def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.
    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.
    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)

def apply_mask(data, mask):
    # masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    masked_data = torch.einsum('hw, nhwc->nhwc', mask, data) + 0.0
    return masked_data

if __name__ == '__main__':
    # deepinv test
    from deepinv.tests.test_physics import test_operators_norm, test_operators_adjointness, test_pseudo_inverse, device

    print('norm test....')
    test_operators_norm('MRI', (2, 320, 320), 'cuda:0') #pass
    print('pinv test....')
    test_pseudo_inverse('MRI', (2, 320, 320), 'cuda:0') #pass
    print('adjoint test....')
    test_operators_adjointness('MRI', (2,320,320), 'cuda:0')#pass, tensor(0., device='cuda:0')

    print('pass all...')

    # norm test....
    # Power iteration converged at iteration 1, value=1.00
    # pinv test....
    # adjoint test....
    # adjoint error= tensor(0., device='cuda:0')
    # pass all...
