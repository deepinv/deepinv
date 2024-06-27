import numpy as np
import torch
import torch.fft
from torch import Tensor
from typing import List, Optional, Union
import warnings
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

    A fixed mask can be set at initialisation, or a new mask can be set either at forward (using ``physics(x, mask=mask)``) or using ``update_parameters``.

    :param torch.Tensor mask: binary mask, where 1s represent sampling locations, and 0s otherwise.
        The mask size can either be (H,W), (C,H,W), or (B,C,H,W) where H, W are the image height and width, C is channels (typically 2) and B is batch size.
    :param torch.device device: cpu or gpu.

    |sep|

    :Examples:

        Single-coil accelerated MRI operator with subsampling mask:

        >>> from deepinv.physics import MRI
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 2, 2, 2) # Define random 2x2 image
        >>> mask = 1 - torch.eye(2) # Define subsampling mask
        >>> physics = MRI(mask=mask) # Define mask at initialisation
        >>> physics(x)
        tensor([[[[ 0.0000, -1.4290],
                  [ 0.4564, -0.0000]],
        <BLANKLINE>
                 [[ 0.0000,  1.8622],
                  [ 0.0603, -0.0000]]]])
        >>> physics = MRI(img_size=x.shape) # No subsampling
        >>> physics(x)
        tensor([[[[ 2.2908, -1.4290],
                  [ 0.4564, -0.1814]],
        <BLANKLINE>
                 [[ 0.3744,  1.8622],
                  [ 0.0603, -0.6209]]]])
        >>> physics.update_parameters(mask=mask) # Update mask on the fly
        >>> physics(x)
        tensor([[[[ 0.0000, -1.4290],
                  [ 0.4564, -0.0000]],
        <BLANKLINE>
                 [[ 0.0000,  1.8622],
                  [ 0.0603, -0.0000]]]])


    """

    def __init__(
        self,
        mask: Optional[Tensor] = None,
        img_size: Optional[tuple] = (320, 320),
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.img_size = img_size

        if mask is None:
            mask = torch.ones(*img_size)

        self.update_parameters(mask=mask.to(self.device))

    def V_adjoint(self, x: Tensor) -> Tensor:  # (B, 2, H, W) -> (B, H, W, 2)
        y = fft2c_new(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return y

    def U(self, x: Tensor) -> Tensor:
        if not torch.all((self.mask == 0) | (self.mask == 1)):
            warnings.warn("The mask must have values 0 or 1")

        if x.size(0) != self.mask.size(0) and self.mask.size(0) != 1:
            raise ValueError(
                "The batch size of the mask and the input should be the same or the mask batch size must be 1."
            )
        return x * self.mask

    def U_adjoint(self, x: Tensor) -> Tensor:
        return self.U(x)

    def V(self, x: Tensor) -> Tensor:  # (B, 2, H, W) -> (B, H, W, 2)
        x = x.permute(0, 2, 3, 1)
        return ifft2c_new(x).permute(0, 3, 1, 2)

    def update_parameters(self, mask=None, check_mask=True, **kwargs):
        return super().update_parameters(
            mask=self.check_mask(mask=mask) if check_mask else mask, **kwargs
        )

    def check_mask(self, mask: Tensor = None) -> None:
        r"""
        Updates MRI mask and verifies mask shape to be B,C,H,W.

        :param torch.nn.Parameter, float MRI subsampling mask.
        """
        if mask is not None:
            mask = mask.to(self.device)

            while len(mask.shape) < 4:  # to B,C,H,W
                mask = mask.unsqueeze(0)

            if mask.shape[1] == 1:  # make complex if real
                mask = torch.cat([mask, mask], dim=1)

        return mask


class DynamicMRI(MRI):
    r"""
    Single-coil accelerated dynamic magnetic resonance imaging.

    The linear operator operates in 2D+t videos and is defined as

    .. math::

        y_t = S_t Fx_t

    where :math:`S` applies a time-varying mask, and :math:`F` is the 2D discrete Fourier Transform.
    This operator has a simple singular value decomposition, so it inherits the structure of
    :meth:`deepinv.physics.DecomposablePhysics` and thus have a fast pseudo-inverse and prox operators.

    The complex images :math:`x` and measurements :math:`y` should be of size (B, 2, H, W) where the first channel corresponds to the real part
    and the second channel corresponds to the imaginary part.

    A fixed mask can be set at initialisation, or a new mask can be set either at forward (using ``physics(x, mask=mask)``) or using ``update_parameters``.

    :param torch.Tensor mask: binary mask, where 1s represent sampling locations, and 0s otherwise.
        The mask size can either be (H,W), (T,H,W), (C,T,H,W) or (B,C,T,H,W) where H, W are the image height and width, T is time-steps, C is channels (typically 2) and B is batch size.
    :param torch.device device: cpu or gpu.

    |sep|

    :Examples:

        Single-coil accelerated 2D+t MRI operator:

        >>> from deepinv.physics import DynamicMRI
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 2, 2, 2, 2) # Define random video of shape (B,C,T,H,W)
        >>> mask = torch.rand_like(x) > 0.75 # Define random 4x subsampling mask
        >>> physics = DynamicMRI(mask=mask) # Physics with given mask
        >>> physics.update_parameters(mask=mask) # Alternatively set mask on-the-fly
        >>> physics(x)
        tensor([[[[[-0.0000,  0.7969],
                   [-0.0000, -0.0000]],
        <BLANKLINE>
                  [[-0.0000, -1.9860],
                   [-0.0000, -0.4453]]],
        <BLANKLINE>
        <BLANKLINE>
                 [[[ 0.0000,  0.0000],
                   [-0.8137, -0.0000]],
        <BLANKLINE>
                  [[-0.0000, -0.0000],
                   [-0.0000,  1.1135]]]]])

    """

    def flatten(self, a: Tensor) -> Tensor:
        B, C, T, H, W = a.shape
        return a.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    def unflatten(self, a: Tensor, batch_size=1) -> Tensor:
        BT, C, H, W = a.shape
        return a.reshape(batch_size, BT // batch_size, C, H, W).permute(0, 2, 1, 3, 4)

    def A(self, x: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        mask = self.check_mask(self.mask if mask is None else mask)

        mask_flatten = self.flatten(mask.expand(*x.shape)).to(x.device)
        y = self.unflatten(
            super().A(self.flatten(x), mask_flatten, check_mask=False),
            batch_size=x.shape[0],
        )

        self.update_parameters(mask=mask, **kwargs)
        return y

    def A_adjoint(self, y: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        mask = self.check_mask(self.mask if mask is None else mask)

        mask_flatten = self.flatten(mask.expand(*y.shape)).to(y.device)
        x = self.unflatten(
            super().A_adjoint(self.flatten(y), mask=mask_flatten, check_mask=False),
            batch_size=y.shape[0],
        )

        self.update_parameters(mask=mask, **kwargs)
        return x

    def A_dagger(self, y: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        return self.A_adjoint(y, mask=mask, **kwargs)

    def check_mask(
        self,
        mask: torch.Tensor = None,
    ) -> None:
        r"""
        Updates MRI mask and verifies mask shape to be B,C,T,H,W.

        :param torch.nn.Parameter, float MRI subsampling mask.
        """
        while mask is not None and len(mask.shape) < 5:  # to B,C,T,H,W
            mask = mask.unsqueeze(0)

        return super().check_mask(mask=mask)


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
