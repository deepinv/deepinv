import numpy as np
import torch
import torch.fft
from typing import List, Optional
from deepinv.physics.forward import DecomposablePhysics


class MRI(DecomposablePhysics):
    r"""
    Single-coil accelerated magnetic resonance imaging.

    The linear operator operates in 2D slices and is defined as

    .. math::

        y = SFx

    where :math:`S` applies a mask (subsampling operator), and :math:`F` is the 2D discrete Fourier Transform.
    This operator has a simple singular value decomposition, so it inherits the structure of
    :meth:`deepinv.physics.DecomposablePhysics` and thus have a fast pseudo-inverse and prox operators.

    The complex images :math:`x` and measurements :math:`y` should be of size (B, 2, H, W) where the first channel corresponds to the real part
    and the second channel corresponds to the imaginary part.

    :param torch.Tensor mask: binary mask.
        The mask size should be of the form (H,W) where H is the image height and W is the image width.
    :param torch.device device: cpu or gpu.

    |sep|

    :Examples:

        Single-coil MRI operator with 4x acceleration:

        >>> from deepinv.physics import MRI
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 2, 3, 3) # Define random 3x3 image
        >>> mask = torch.ones((3, 3))
        >>> mask[:, ::2] = 0
        >>> physics = MRI(mask=mask)
        >>> physics(x)
        tensor([[-0.5305,  0.0351,  0.3326,  2.1730,  1.7072,  0.0418]])

    """

    def __init__(
        self,
        mask,
        img_size=(320, 320),
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.img_size = img_size

        mask = mask.to(device)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        self.mask = torch.nn.Parameter(
            torch.cat([mask, mask], dim=1), requires_grad=False
        ).to(device)

    def V_adjoint(self, x):  # (B, 2, H, W) -> (B, H, W, 2)
        y = fft2c_new(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return y

    def U(self, x):
        return x[:, self.mask.squeeze(0) > 0]

    def U_adjoint(self, x):
        _, c, h, w = self.mask.shape
        out = torch.zeros((x.shape[0], c, h, w), device=x.device)
        out[:, self.mask.squeeze(0) > 0] = x
        return out

    def V(self, x):  # (B, 2, H, W) -> (B, H, W, 2)
        x = x.permute(0, 2, 3, 1)
        return ifft2c_new(x).permute(0, 3, 1, 2)


#
# reference: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    r"""
    Apply centered 2 dimensional Fast Fourier Transform.
    :param torch.Tensor data: Complex valued input data containing at least 3 dimensions:
        dimensions -2 & -1 are spatial dimensions and dimension -3 has size
        2. All other dimensions are assumed to be batch dimensions.
    :param bool norm: Normalization mode. See ``torch.fft.fft``.
    :return: (torch.tensor) the FFT of the input.
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


def roll(x: torch.Tensor, shift: List[int], dim: List[int]) -> torch.Tensor:
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

    for s, d in zip(shift, dim):
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


# if __name__ == "__main__":
#     # deepinv test
#     from deepinv.tests.test_physics import (
#         test_operators_norm,
#         test_operators_adjointness,
#         test_pseudo_inverse,
#         device,
#     )
#     import deepinv as dinv
#     from fastmri.data import subsample
#
#     imsize = (25, 32)
#     # Create a mask function
#     mask_func = subsample.RandommaskFunc(center_fractions=[0.08], accelerations=[4])
#     m = mask_func.sample_mask((imsize[1], imsize[0]), offset=None)
#
#     # mask = torch.ones((imsize[0], 1)) * (m[0] + m[1]).permute(1, 0)
#     mask = torch.ones(imsize)
#     mask[mask > 1] = 1
#
#     sigma = 0.1
#     # physics = MRI(mask=mask, device=dinv.device)
#     physics = dinv.physics.Denoising()
#     physics.noise_model = dinv.physics.GaussianNoise(sigma)
#
#     # choose a reconstruction architecture
#     backbone = dinv.models.MedianFilter()
#
#     class denoiser(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#
#         def forward(self, x, sigma=None):
#             return x
#
#     f = dinv.models.ArtifactRemoval(backbone)
#
#     batch_size = 1
#
#     for tau in np.logspace(-5, 3, 1):
#         x = torch.ones((batch_size, 2) + imsize, device=dinv.device)
#         y = physics(x)
#
#         # choose training losses
#         loss = dinv.loss.SureGaussianLoss(sigma, tau=tau)
#         x_net = f(y, physics)
#         mse = dinv.metric.mse()(physics.A(x), physics.A(x_net))
#         sure = loss(y, x_net, physics, f)
#
#         print(f"tau:{tau:.2e}  mse: {mse:.2e}, sure: {sure:.2e}")
#         rel_error = (sure - mse).abs() / mse
#         print(f"rel_error: {rel_error:.2e}")
#
#     d = physics.A_adjoint(y)
#     dinv.utils.plot([d.sum(1).unsqueeze(1), x.sum(1).unsqueeze(1)])
#
#     print("adjoint test....")
#     test_operators_adjointness(
#         "MRI", (2, 320, 320), dinv.device
#     )  # pass, tensor(0., device='cuda:0')
#     print("norm test....")
#     test_operators_norm("MRI", (2, 320, 320), dinv.device)  # pass
#     print("pinv test....")
#     test_pseudo_inverse("MRI", (2, 320, 320), dinv.device)  # pass
#
#     print("pass all...")
